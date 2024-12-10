from .base import *
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, QueryKeywordsEncoder
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class ETA(BaseSearchBasedModel):
    def __init__(self, args):
        super().__init__(args)
        self.hash_bits = args.gsu_embd_hidden_size
        self.hash_proj_matrix = nn.Linear(args.hidden_size, self.hash_bits)
        # freeze hash proj matrix after initialization
        for param in self.hash_proj_matrix.parameters():
            param.requires_grad = False
            
    def hash_emb_layer(self, inputs):
        """
            inputs:  dense embedding of [B, ..., D]
            inputs_proj_hash: int (0/1) embedding of [B, ..., N_Bits], larger distance means similar vectors
        """
        # [B, ..., D] -> [B, ..., N_bits]
        inputs_proj = self.hash_proj_matrix(inputs)
        inputs_proj = torch.unsqueeze(inputs_proj, dim = -1) # [B, N_Bit] -> [B, N_Bit, 1]
        inputs_proj_merge = torch.cat([-1.0 * inputs_proj, inputs_proj], axis=-1)  # [B, N_Bit, 1] -> [B, N_Bit, 2]
        # 一种转为0、1的方法
        inputs_proj_hash = torch.argmax(inputs_proj_merge, dim=-1)
        return inputs_proj_hash

    def hamming_distance(self, query_hashes, keys_hashes):
        """
            query_hashes: [B, 1, N_Bits]
            keys_hashes: [B, L, N_Bits]
            distance: [B, L]
        """
        key_num = keys_hashes.shape[1]
        # [B, 1, N] -> [B, L, N]
        query_hashes_tile = query_hashes.repeat((1, key_num, 1))
        match_buckets = torch.eq(query_hashes_tile, keys_hashes).int()
        distance = torch.sum(match_buckets, dim=-1)
        return distance
    
    
    def stageone_softsearch(self,user_seq_emb,target_emb):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]

            output:
                gsu_out_topk     [B, K]
                gsu_merged   [B, D]  Stage1 Merged User Representation, Pass the tensor to stage 1 loss

            功能：
                利用target_emb在user_seq_emb search出最接近的K个item,返回id而不是embedding
        """
        batch_size, sequence_len, emb_dim = user_seq_emb.shape
        query_hashes = self.hash_emb_layer(target_emb)
        keys_hashes = self.hash_emb_layer(user_seq_emb)
        # ETA Caculate Hamming Distance, [B, L]
        qk_hamming_distance = self.hamming_distance(query_hashes, keys_hashes)

        # values: [B, K], indices: [B, K]
        values, indices = torch.topk(qk_hamming_distance, K, dim=-1, largest=True)
        
        return indices,None
        
    
    
    def item_id2logit(self,user_sequence_id,target_item_id):
        """
            input:
                user_sequence_emb: [B, L] 
                target_emb: [B, 1]

            output:
                logit_esu: [B, 1]
                logit_gsu: [B, 1]
                
            功能：
                将id转为embedding,并拼接sequence_emb与target_emb
        """
        user_seq_embd , _ , target_item_embd = self.item_embed_layer(user_sequence_id,target_item_id)
        
        pos_relative_gsu_out_topk , _ = self.stageone_softsearch(user_seq_embd , target_item_emb)
        esu_output_pos, _ = self.stagetwo_embdfusion(pos_relative_gsu_out_topk ,user_seq_embd ,  target_item_emb)
        # torch.squeeze可以干掉张量中所有维度为1的shape
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )
        
        return logit_pos_esu , None
    