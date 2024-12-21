import torch
import wandb
import argparse

from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *


def train(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + '/' + args.desc

    train, val, test = dataloader_factory(args)
    model = model_factory(args)
    trainer = BaseTrainer(args, model, train, val, test, export_root, args.use_wandb)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    config = load_config(args.config)
    train(config)
