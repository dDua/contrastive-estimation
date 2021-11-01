# Contrastive Estimation

The main file is ropes_ablations.py where the class name for the model can be provided in model_class variable

To run a specific CE model, change the datafile paths and pretrained asnwering model path in configs/ropes_config.py.
Then start training the CE model with

python3 -m torch.distributed.launch --nproc_per_node=4 ropes_ablations.py --model_checkpoint <pretrained answering path> --output_dir <output_path>
