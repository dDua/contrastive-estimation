# Contrastive Estimation

The main file is ropes_ablations.py where the class name for the model can be provided in model_class variable

To run a specific CE model, change the datafile paths and pretrained answering model path in configs/ropes_config.py.
Then start training the CE model with

python3 -m torch.distributed.launch --nproc_per_node=4 ropes_ablations.py --model_checkpoint <pretrained answering model path> --output_dir <output_path>

If it is a new dataset and you don't have an answering model for it, then it can be trained by passing loss type as only "mle" while initializing model_class to ContrastiveEstimationAnswerCond type model

This code requires following dependencies

+ python3.6 
+ transformers (version 2.9.1)
+ pytorch-ignite (version 0.2.0)
