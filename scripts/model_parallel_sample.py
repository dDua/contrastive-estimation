import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import torch.nn as nn
import os
import torch.optim as optim

cuda2 = torch.device('cuda:2')
cuda3 = torch.device('cuda:3')

def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '134358'

    # initialize the process group
    dist.init_process_group("nccl", rank=0, world_size=1)

def cleanup():
    dist.destroy_process_group()

class MyModel(nn.Module):
    def __init__(self, embeddings):
        super().__init__()

        self.model1 = nn.Sequential()
        self.model1.add_module('ctx', embeddings)
        self.model1.add_module('lin2', nn.Linear(2000,1000))
        self.model1.add_module('act', nn.ReLU())
        self.model1.add_module('lin3', nn.Linear(1000,1))

        self.model2 = nn.Sequential()
        self.model2.add_module('ctx', embeddings)
        self.model2.add_module('lin2', nn.Linear(2000, 500))
        self.model2.add_module('act', nn.ReLU())
        self.model2.add_module('lin3', nn.Linear(500, 1))

       # self.register_buffer("embeddings", embeddings)
        self.model1 = self.model1.to(cuda2)
        self.model2 = self.model2.to(cuda3)

    def forward(self, data):
        data1 = data.to(cuda2)
        output1 = self.model1(data1).squeeze(-1).sum(-1)
        data2 = data.clone().to(cuda3)
        output2 = self.model2(data2).squeeze(-1).sum(-1)
        output = output1 + output2.to(cuda2)
        return output

def demo_model_parallel(args):
    setup()
    data = torch.randint(0, 29, (100, 200))
    labels = torch.randn(100)

    embeddings = nn.Sequential()
    embeddings.add_module('emb', nn.Embedding(30,1000))
    embeddings.add_module('line', nn.Linear(1000,2000))

    # setup mp_model and devices for this process
    model = MyModel(embeddings)
    model.share_memory()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = model(data)
    print(outputs)
    labels = labels.to(cuda2)
    loss = loss_fn(outputs, labels)
    loss.backward()
    print(loss)
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size=1):
    mp.spawn(demo_fn,
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(demo_model_parallel)
