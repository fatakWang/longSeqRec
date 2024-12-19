from .base import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class DIN(BaseSearchBasedModel):
    def __init__(self, args):
        super().__init__(args)
    
    def stageone_softsearch(self,user_seq_emb,target_emb):
        return user_seq_emb,None
    
    def stagetwo_embdfusion(self,gsu_out_topk,user_seq_emb ,target_emb):
        gsu_out_topk = user_seq_emb[:,-self.args.K:,:]
        
        num_heads = self.args.num_heads
        head_list = []
        emb_dim = user_seq_emb.shape[-1]

        attention_mask_list = []
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            sequence_emb_proj = self.wK_list[i](gsu_out_topk)
            value_emb_proj = self.wV_list[i](gsu_out_topk)
            # [B, L]
            attention_mask = torch.softmax(
                torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(sequence_emb_proj, 1, 2))) / np.sqrt(emb_dim),
                dim=-1)
            attention_mask_list.append(attention_mask)
            # attention_mask: [B,L] value_emb_proj: [B, L, D] ->  [B, D, L]  [B, L, 1] -> [B, D, 1] -> [B, D]
            head_out = torch.squeeze(
                torch.bmm(torch.transpose(value_emb_proj, 1, 2), torch.unsqueeze(attention_mask, -1)))
            # print("DEBUG: The %d-th head shape %s" % (i, str(head_out.shape)))
            head_list.append(head_out)
        mhta_output = self.wO(torch.cat(head_list, dim=-1))
        return mhta_output, attention_mask_list