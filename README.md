# contrastive-estimation

### Quoref test command
```
# Uses parameters from configs/t5_quoref_config.py

python run_quoref_answering_model.py

# For multi-gpu training
python -m torch.distributed.launch --nproc_per_node=8 run_quoref_answering_model.py  
```

### Conda env create
```
conda create --prefix /shared/nitishg/envs/ce python=3.6

pip install allennlp==1.0.0

# Pytorch 1.5.1
# CUDA 10.2
pip install torch==1.5.1 torchvision==0.6.1
# CUDA 10.1
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html


pip install pytorch-ignite==0.2.0
pip install tensorflow-gpu==1.12.0
conda install cudatoolkit=9.0
conda install -c anaconda cudnn # (v7.6.5)
pip install transformers==2.9.1

pip install word2number==1.1

pip install cython
pip install benepar==0.1.2
# python -> import nltk -> import benepar -> 
# benepar.download('benepar_en2')

pip install spacy==2.1.3
python -m spacy download en_core_web_sm

pip install fasttext==0.9.2
pip install lemminflect==0.2.1

```