import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import multiprocessing
import time

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"

    seed = 1334
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        print_network(model)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        # print('==================')
        while not t.terminate():
            # print('======++++++++++')
            t.train()
            t.test()
        checkpoint.done()
    



