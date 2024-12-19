from .sim import SIM
from .din import DIN
from .eta import ETA
from .twin import TWIN
from .sdim import SDIM
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torchinfo import summary
MODELS = {
    "DIN": DIN,
    "SIM": SIM,
    "ETA": ETA,
    "TWIN": TWIN,
    "SDIM": SDIM
    
}

def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)

class test():
    def __init__(self):
        self.model_code = "SDIM"
        self.item_size = 100
        self.hidden_size = 64
        self.gsu_embd_hidden_size = 2
        self.num_heads = 4
        self.K = 2
        self.device = "cpu"
        

if __name__ == "__main__":
    args = test()
    B,L = 64,100
    input_sequence_ids = torch.randint(0,args.item_size,(B,L)).to(args.device)
    target_item_id_pos = torch.randint(0,args.item_size,(B,1)).to(args.device)
    target_item_id_neg = torch.randint(0,args.item_size,(B,101)).to(args.device)
    
    model = model_factory(args).to(args.device)
    # model.item_embed_layer(input_sequence_ids,target_item_id)
    # model.stageone_softsearch()
    # model.stagetwo_embdfusion
    # model.prediction_layer
    # model.item_id2logit(input_sequence_ids,target_item_id_pos)
    model.get_logit_pos_neg(input_sequence_ids,target_item_id_pos,target_item_id_neg)
    # model.test_search(input_sequence_ids,target_item_id_pos)
    # summy