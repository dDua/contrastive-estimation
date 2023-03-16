# Adapted from original codebase written by HuggingFace Inc. (https://github.com/huggingface/transfer-learning-conv-ai)

import os
import math
import logging
import random
from pprint import pformat
import json
from multiprocessing import Queue
from queue import Empty
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, MeanAbsoluteError
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW
from model.contrastive_models import *
from configs.comparison_config import get_arguments as get_arguments_comp
from data.data_process_hotpot import *
from utils import get_exact_match, get_f1

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def train():
    queue = Queue()
    args, logger = get_arguments_comp()
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")

    tokenizer_class, model_class, dataset_class = T5Tokenizer, ContrastiveEstimationQuestionCond, \
                                                  HotpotQAData


    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                   "cls_token": "<cls>", "additional_special_tokens": dataset_class.special_tokens})

    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy, y_only=False, x_only=False, y_types='mine',
                            x_types='gen')
    out_symbol_idx = tokenizer.convert_tokens_to_ids("<answer>")

    model = model_class.from_pretrained(args.model_checkpoint, **{"ans_sym_id": out_symbol_idx,
                                                                  "max_ans_len": args.max_output_length,
                                                                  "tokenizer": tokenizer,
                                                                  "loss_type": ["mle", "lnorm"]})
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                        find_unused_parameters=True)

    logger.info("Prepare datasets")
    train_loader, train_sampler, val_loader, valid_sampler = dataset.get_data_loaders(lazy=args.lazy)
    print(len(train_loader), len(val_loader))

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if len(batch) == 5:
            input_ids, attention_mask, output_src, output_tgt, output_mask = batch
            contrast_labels=None
        else:
            input_ids, attention_mask, output_src, output_tgt, output_mask, contrast_labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=output_src,
                        lm_labels=output_tgt, decoder_attention_mask=output_mask, contrast_labels=contrast_labels)

        loss = outputs[0] / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                  scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        eos_symbol = tokenizer.convert_tokens_to_ids("<eos>")

        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

            input_ids, attention_mask, output_src, output_tgt, output_mask = batch

            batch_size, num_samples_q, seq_len = input_ids.size()
            _, _, ans_len = output_src.size()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=output_src,
                        lm_labels=output_tgt, decoder_attention_mask=output_mask, max_len=20,
                        generate_answer=True)
            generate_indices, generated_probs = outputs[0], outputs[1]

            answer_logits_masked = torch.masked_select(generated_probs.reshape(-1, generated_probs.size(-1)),
                                               output_mask.reshape(-1, 1).bool()).view(-1, generated_probs.size(-1))
            answer_labels_masked = torch.masked_select(output_tgt.reshape(-1),
                                                       output_mask.reshape(-1).bool())

            generate_indices = generate_indices.tolist()
            output_src[output_src == -100] = 0
            output_tgt[output_tgt == -100] = 0
            em, f1 = [], []
            for b in range(batch_size):
                original_answer_k = tokenizer.decode(output_tgt[b][0].tolist(), skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True).lower().strip()
                if eos_symbol in generate_indices[b][0]:
                    out_end_len = generate_indices[b][0].index(eos_symbol)
                else:
                    out_end_len = -1
                generated_answer_k = tokenizer.decode(generate_indices[b][0][:out_end_len], skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True).lower().strip()
                em.append(get_exact_match(generated_answer_k, original_answer_k))
                f1.append(get_f1(generated_answer_k, original_answer_k))


            if original_answer_k.strip() and random.uniform(0, 1) <= args.output_prob:
                queue.put((json.dumps(generated_answer_k), json.dumps(original_answer_k)))

            return (answer_logits_masked, torch.tensor(em).long().view(-1), 1+torch.tensor(f1).view(-1)), \
                   (answer_labels_masked, torch.ones(batch_size, 1).view(-1))

    def queue_reader(foo):
        while True:
            try:
                print(queue.get_nowait())
            except Empty:
                break

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, queue_reader)

    # if args.n_epochs < 1:
    #     trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {
        "ans_accuracy": Accuracy(output_transform=lambda x: (x[0][0], x[1][0])),
        "em": Accuracy(output_transform=lambda x: (x[0][1], x[1][1])),
        "f1": MeanAbsoluteError(output_transform=lambda x: (x[0][2], x[1][1]))
    }

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=args.output_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        try:
            tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
        except Exception:
            def global_step_transform(*args, **kwargs):
                return trainer.state.epoch
            tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                                  global_step_transform=global_step_transform),
                             event_name=Events.EPOCH_COMPLETED)

        def scoring_fn(engine):
            return engine.state.metrics["f1"]

        checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', n_saved=2, require_empty=False,
                                             score_function=scoring_fn)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
                                    {'mymodel': getattr(model, 'module', model)})
        # "getattr" take care of distributed encapsulation

        torch.save(args, args.output_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.output_dir, "config.json"))
        tokenizer.save_vocabulary(args.output_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.output_dir, "model_training_args.bin"))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
