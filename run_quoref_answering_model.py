# Copyright (c) 2019-present
# Original codebase written by HuggingFace Inc. (https://github.com/huggingface/transfer-learning-conv-ai)

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
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW
from model.answering_model import T5QA
from configs.t5_ropes_config import get_arguments as get_arguments_ropes
from configs.t5_quoref_config import get_arguments as get_arguments_quoref
from data.data_processing_quoref import QuorefQADataBaseline
from data.data_processing_torque import TorqueQADataBaseline
from configs.t5_torque_config import get_arguments as get_arguments_torque

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def train():
    args, logger = get_arguments_torque()
    queue = Queue()
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

    tokenizer_class, model_class, dataset_class = T5Tokenizer, T5QA, QuorefQADataBaseline

    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                   "cls_token": "<cls>", "additional_special_tokens": dataset_class.special_tokens})
    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy, aug=False)
    model = model_class.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
                                                                "max_ans_len": args.max_output_length,
                                                                "tokenizer": tokenizer
                                                                })
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

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, attention_mask, answer_input, answer_output, answer_mask, _ = batch

        losses = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=answer_input,
                        lm_labels=answer_output, decoder_attention_mask=answer_mask, entity_type_ids=None)
        answer_loss = losses[-1]
        loss = answer_loss / args.gradient_accumulation_steps

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
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

            input_ids, attention_mask, answer_input, answer_output, answer_mask, _ = batch

            pos_input_ids = input_ids[:, 0, :].unsqueeze(1)
            pos_attention_mask = attention_mask[:, 0, :].unsqueeze(1)
            hidden = model(pos_input_ids, pos_attention_mask, encode_only=True)
            generated_ans_indices, generated_probs = model(pos_input_ids, attention_mask=pos_attention_mask,
                            encoded_hidden_states=hidden, max_len=answer_mask.size(1))

            ans_labels = answer_output.contiguous().view(-1)

            answer_logits_masked = torch.masked_select(generated_probs.reshape(-1, generated_probs.size(-1)),
                                                       answer_mask.reshape(-1,1).bool()).view(-1, generated_probs.size(-1))
            answer_labels_masked = torch.masked_select(ans_labels, answer_mask.view(-1).bool())

            if random.uniform(0, 1) <= args.output_prob:

                generated_ans_indices = generated_ans_indices.tolist()
                gold_ans_indices = answer_input.tolist()
                try:
                    gen_eos_idx = generated_ans_indices[0].index("<eos>")
                except Exception:
                    gen_eos_idx = -1

                try:
                    gold_eos_idx = gold_ans_indices[0].index(-100)
                except Exception:
                    gold_eos_idx = -1

                generated_answer = tokenizer.decode(generated_ans_indices[0][:gen_eos_idx], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
                answer_input[answer_input == -100] = 0
                original_answer = tokenizer.decode(gold_ans_indices[0][:gold_eos_idx], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True)

                queue.put((json.dumps(generated_answer), json.dumps(original_answer)))

            return (answer_logits_masked, answer_labels_masked)

    evaluator = Engine(inference)


    def queue_reader(foo):
        while True:
            try:
                print(queue.get_nowait())
            except Empty:
                break

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, queue_reader)

    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

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
        "ans_accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))
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
        checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', save_interval=1, n_saved=15, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

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
