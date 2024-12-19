from .base import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class TWIN(BaseSearchBasedModel):
    def __init__(self, args):
        super().__init__(args)
        
    
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
        user_sequence_emb , _ , target_item_emb = self.item_embed_layer(user_sequence_id,target_item_id)
        
        gsu_output_topK_emb_list , _= self.stageone_softsearch(user_sequence_emb , target_item_emb)
        esu_output_pos, _ = self.multi_head_target_attention(gsu_output_topK_emb_list ,  target_item_emb)
        # torch.squeeze可以干掉张量中所有维度为1的shape
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )
        # logit_pos_gsu_merged = self.prediction_layer(pos_gsu_merged ,torch.squeeze(target_item_emb) )
        
        return logit_pos_esu , None
    
    def stageone_softsearch(self,user_seq_emb,target_emb):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]

            output:
                gsu_output_topK_emb_list     [B, K,D,H]
                gsu_merged   [B, D]  Stage1 Merged User Representation, Pass the tensor to stage 1 loss

            功能：
                利用target_emb在user_seq_emb search出最接近的K个item,返回id而不是embedding
        """
        num_heads = self.args.num_heads
        emb_dim = user_seq_emb.shape[-1]

        attention_list = []
        gsu_output_topK_emb_list = [] # list of [B, K, D]
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            user_seq_emb_proj = self.wK_list[i](user_seq_emb)
            # value_emb_proj = self.wV_list[i](user_seq_emb)
            # [B, L]
            # attention_mask = torch.softmax(
            #     torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(sequence_emb_proj, 1, 2))) / np.sqrt(emb_dim),
            #     dim=-1)
            qK = torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(user_seq_emb_proj, 1, 2)))  # [B,L,D] \times [B, D, 1] -> [B, L, 1] -> [B, L]
            attention = torch.softmax(qK/np.sqrt(emb_dim), dim=-1)
            attention_list.append(attention)
            # Current Attention Mask Cut Top K
            values, indices = torch.topk(qK, self.args.K, dim=-1, largest=True)
            # user_seq_emb: [B, L, D], index= [B, index_length=K]
            gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64)
            # user_seq_emb [B, L, D] -> [B, K, D]
            gather_topk_emb = torch.gather(user_seq_emb, dim=1, index=gather_index, out=None)
            gsu_output_topK_emb_list.append(gather_topk_emb)
        return gsu_output_topK_emb_list, None
    
    
    def multi_head_target_attention(self, sequence_emb_list, target_emb):
        """
            input:
                sequence_emb_list: [B, L, D] list size num_head
                target_emb: [B, 1, D]

            output:
                mhta_output: [B, D]
                attention_mask_list: list of attention_mask [B, L]
        """
        num_heads = self.args.num_heads
        head_list = []
        emb_dim = sequence_emb_list[0].shape[-1]

        attention_mask_list = []
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            sequence_emb_proj = self.wK_list[i](sequence_emb_list[i])
            value_emb_proj = self.wV_list[i](sequence_emb_list[i])
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