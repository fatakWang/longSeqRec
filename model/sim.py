from .base import *
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, QueryKeywordsEncoder
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class SIM(BaseSearchBasedModel):
    def __init__(self, args):
        super().__init__(args)
        self.esu_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.gsu_embeddings = nn.Embedding(args.item_size, args.gsu_embd_hidden_size, padding_idx=0)
        self.gsu_merged_out_alignment_layer = nn.Linear(args.gsu_embd_hidden_size,args.hidden_size)
        del self.item_embeddings
        
        
    def item_embed_layer(self,input_sequence_ids,target_item_id):
        """
            input:
                input_sequence_ids: shape: [B, L] type: int 
                target_item_id: shape: [B, 1] type: int 

            output:
                gsu_embd = [B, L, D]
                esu_embd = [B, L, D]
                target_item_gsu_embd = [B, 1, D]
                target_item_esu_embd = [B, 1, D]
                
            功能：
                物品嵌入层，将L个item的id转为embedding,
        """
        gsu_embd = self.gsu_embeddings(input_sequence_ids)
        esu_embd = self.esu_embeddings(input_sequence_ids)
        target_item_gsu_embd = self.gsu_embeddings(target_item_id)
        target_item_esu_embd = self.esu_embeddings(target_item_id)
        return gsu_embd , esu_embd , target_item_gsu_embd , target_item_esu_embd
    
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
        gsu_embd , esu_embd , target_item_gsu_embd , target_item_esu_embd = self.item_embed_layer(user_sequence_id,target_item_id)
        
        pos_relative_gsu_out_topk , pos_gsu_merged= self.stageone_softsearch(gsu_seq_embd , target_item_gsu_embd)
        esu_output_pos, _ = self.stagetwo_embdfusion(pos_relative_gsu_out_topk ,esu_seq_embd ,  target_item_esu_embd)
        # torch.squeeze可以干掉张量中所有维度为1的shape
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )
        # [B,args.gsu_embd_hidden_size] -> [B,hidden_size]
        pos_gsu_merged = self.gsu_merged_out_alignment_layer(pos_gsu_merged)
        logit_pos_gsu_merged = self.prediction_layer(pos_gsu_merged ,torch.squeeze(target_item_emb) )
        
        return logit_pos_esu , logit_pos_gsu_merged
        
    
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
        # 用target_emb与user_seq_emb做内积search出前K个
        qK = torch.squeeze(torch.bmm(user_seq_emb, torch.transpose(target_emb, 1, 2)))   # [B,L,D] \times [B, D, 1] -> [B, L, 1] -> [B, L]
        # values: [B, K], indices: [B, K]，他不怕会选出0么
        # TODO 后续需要确保不会search出padding item
        values, indices = torch.topk(qK, self.args.K, dim=-1, largest=True)

        ## SIM merge, [B, L, D], [B, L, 1] -> [B, D] ， gsu_out_merge并没有QKv的过程也没有scaled,相当于qk内积后直接得到的是重要度然后与原序列emb相乘，得到融合后的结果
        # 这个矩阵相乘的过程相当于加权求和
        gsu_out_merge = torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(qK, dim=-1)).squeeze()

        return indices, gsu_out_merge
        
        