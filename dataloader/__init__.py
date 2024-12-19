from datasets import dataset_factory
from .config import *
from .sas import *


def dataloader_factory(args):
    # 
    dataset = dataset_factory(args)
    dataloader = SASDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test

if __name__ == "__main__":
    args = load_config("C:\\D\code\\longSeqRec\\config.yaml")
    train, val, test = dataloader_factory(args)
    for i in test:
        seq , target_pos , target_neg = i
    # 12月14日15点47分完全解决了        
        # print(f"{seq.shape} {target_pos.shape} {target_neg.shape}")
        pass
    # for i in train:
    #     seq , target_pos , target_neg = i
    #     # print(f"{seq.shape} {target_pos.shape} {target_neg.shape}")
    #     pass
    # for i in val:
    #     seq , target_pos , target_neg = i
    #     # print(f"{seq.shape} {target_pos.shape} {target_neg.shape}")
    #     pass
    # dataset = dataset_factory(args)
    # dataloader = SASDataloader(args, dataset)
    # train_dataset = dataloader._get_train_dataset()
    # for i in train_dataset:
    #     seq, target_pos,target_neg = i
    #     print(f"1  {seq.shape} {target_pos.shape} {target_neg.shape}")