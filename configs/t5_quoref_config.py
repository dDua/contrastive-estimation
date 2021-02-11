import os
import argparse
import logging

import random
import numpy as np
import torch

def get_arguments():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--dataset_path", type=str, default="datasets/quoref/")
    parser.add_argument("--dataset_cache", default="datasets/quoref/cache/")
    parser.add_argument("--predict_file", default="")
    parser.add_argument("--output_dir", default="/tmp/quoref-test", type=str)
    parser.add_argument("--qdmr_path", type=str, default="/mnt/750GB/data/Break-dataset/QDMR-high-level/train.csv")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--train_split_name", default="demo_train")
    parser.add_argument("--dev_split_name", default="demo_dev")
    # parser.add_argument("--train_split_name", default="quoref-train-v0.1")
    # parser.add_argument("--dev_split_name", default="quoref-dev-v0.1")
    parser.add_argument("--distributed", action='store_true', default=True)
    parser.add_argument("--lazy", default=False)
    parser.add_argument("--input_type", type=str, default="Q")

    ## Model parameters
    # parser.add_argument("--model_checkpoint", type=str, default="/mnt/750GB/data/ropes/ropes_answering_model_large_v2")
    # parser.add_argument("--model_checkpoint", type=str, default="/mnt/750GB/data/ropes/ropes_base_2/")
    parser.add_argument("--model_checkpoint", type=str, default="t5-small")
    parser.add_argument("--lowercase", action='store_true', default=False)
    parser.add_argument("--ans_coef", type=float, default=1.0)
    parser.add_argument("--qg_coef", type=float, default=1.0)
    parser.add_argument("--nce_coef", type=float, default=5.0)
    parser.add_argument("--prior_coef", type=float, default=1.0)
    parser.add_argument("--z_coef", type=float, default=1.0)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_question_length', type=int, default=50)
    parser.add_argument('--max_context_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--num_negative", default=5, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--n_qdmr_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--n_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=30, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument('--wait_step', type=int, default=10)
    parser.add_argument(
        # "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
        "--device", type=str, default="cpu"
    )
    parser.add_argument("--fp16", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--reasoning_file",
                        default="/home/ddua/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/reasoning.json")

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--output_prob', type=float, default=0.05)
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    logger.info("Using {} gpus".format(args.n_gpu))

    return args, logger


