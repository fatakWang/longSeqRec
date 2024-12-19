from .base import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class ETA(BaseSearchBasedModel):
    def __init__(self, args):
        super().__init__(args)
        self.hash_bits = args.gsu_embd_hidden_size
        self.hash_proj_matrix = torch.randn(self.args.hidden_size, self.hash_bits, device=self.args.device)
        self.hash_proj_matrix.requires_grad = False 
        
    # 只有训练会用
    def hash_emb_layer(self, inputs):
        """
            inputs:  dense embedding of [B, ..., D]
            inputs_proj_hash: int (0/1) embedding of [B, ..., N_Bits], larger distance means similar vectors
        """
        hash_scores = torch.matmul(inputs, self.hash_proj_matrix)
        hash_values = torch.sign(hash_scores)
        hash_values = (hash_values + 1) / 2
        hash_values = hash_values.type(torch.uint8)
        
        return hash_values
        
    # 测试中只有它
    def hamming_distance(self, query_hashes, keys_hashes):
        """
            query_hashes: [B, 1, N_Bits]
            keys_hashes: [B, L, N_Bits]
            distance: [B, L]
        """
        
        # key_num = keys_hashes.shape[1]
        # # [B, 1, N] -> [B, L, N]
        # query_hashes_tile = query_hashes.repeat((1, key_num, 1))
        # match_buckets = torch.eq(query_hashes_tile, keys_hashes).int()
        # distance = torch.sum(match_buckets, dim=-1)
        match_buckets = torch.bitwise_xor(query_hashes,keys_hashes)
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
        values, indices = torch.topk(qk_hamming_distance, self.args.K, dim=-1,largest=False)
        
        return indices,None
    
    def sim_search(self,user_seq_emb,target_emb):
        # 用target_emb与user_seq_emb做内积search出前K个
        qK = torch.squeeze(torch.bmm(user_seq_emb, torch.transpose(target_emb, 1, 2)))   # [B,L,D] \times [B, D, 1] -> [B, L, 1] -> [B, L]
        # values: [B, K], indices: [B, K]，他不怕会选出0么
        # TODO 后续需要确保不会search出padding item
        values, indices = torch.topk(qK, self.args.K, dim=-1, largest=True)

        ## SIM merge, [B, L, D], [B, L, 1] -> [B, D] ， gsu_out_merge并没有QKv的过程也没有scaled,相当于qk内积后直接得到的是重要度然后与原序列emb相乘，得到融合后的结果
        # 这个矩阵相乘的过程相当于加权求和
        gsu_out_merge = torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(qK, dim=-1)).squeeze()

        return indices, gsu_out_merge
    
    
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
        user_seq_embd , _ , target_item_emb = self.item_embed_layer(user_sequence_id,target_item_id)
        
        pos_relative_gsu_out_topk , _ = self.stageone_softsearch(user_seq_embd , target_item_emb)
        esu_output_pos, _ = self.stagetwo_embdfusion(pos_relative_gsu_out_topk ,user_seq_embd ,  target_item_emb)
        # torch.squeeze可以干掉张量中所有维度为1的shape
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )
        
        return logit_pos_esu , None
    
    def test_search(self,user_sequence_id,target_item_id):
        # 测试通过，没什么问题
        user_seq_embd , _ , target_item_emb = self.item_embed_layer(user_sequence_id,target_item_id)
        
        indices_simhash , _ = self.stageone_softsearch(user_seq_embd , target_item_emb)
        indices_inner,_ = self.sim_search(user_seq_embd , target_item_emb)
        print(f"{indices_simhash - indices_inner}")
        
    