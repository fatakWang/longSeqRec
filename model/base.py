import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, QueryKeywordsEncoder
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class BaseSearchBasedModel(nn.Module):
    """
        Baseline Model
    """
    def __init__(self, args):
        super(BaseSearchBasedModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.final_deep_layers = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
        )
    
    # 对用户行为序列、item 得到嵌入、
    def item_embed_layer(self,input_sequence_ids):
        """
            input:
                input_sequence_ids: shape: [B, L] type: int 

            output:
                [B, L, D]
                
            功能：
                物品嵌入层，将L个item的id转为embedding,
        """
        item_id_embeddings = self.item_embeddings(input_sequence_id)
        return item_id_embeddings

        
    def stageone_softsearch(self,user_seq_emb,target_emb):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]

            output:
                gsu_out_topk     [B, K, D]
                gsu_merged   [B, D]  Stage1 Merged User Representation, Pass the tensor to stage 1 loss

            功能：
                利用target_emb在user_seq_emb search出最接近的K个item
        """
        # 交给子类去实现
        raise NotImplementedError
    
    def stageone_allinone_softsearch(self,user_seq_emb,target_emb):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, candidate_size, D]

            output:
                gsu_out_topk     [B, candidate_size , K, D]
                gsu_merged   [B, candidate_size , D]  Stage1 Merged User Representation, Pass the tensor to stage 1 loss

            功能：
                利用target_emb在user_seq_emb search出最接近的K个item
        """
        # 交给子类去实现
        raise NotImplementedError
        
    def stagetwo_embdfusion(self,sequence_emb,target_emb):
        """
            input:
                sequence_emb_list: [B, K, D] 
                target_emb: [B, 1, D]

            output:
                mhta_output: [B, D]
                attention_mask_list: list of attention_mask [B, L]
                
            功能：
                将sequence_emb融合为一个定长的embedding
        """
        # 交给子类去实现
        raise NotImplementedError 
    
    def stagetwo_allinone_embdfusion(self,user_seq_emb,target_emb):
        """
            input:
                sequence_emb_list: [B, candidate_size , K, D] 
                target_emb: [B , candidate_size, D]

            output:
                mhta_output: [B, candidate_size ,D]
                attention_mask_list: list of attention_mask [B, candidate_size, L]
                
            功能：
                将sequence_emb融合为一个定长的embedding
        """
        # 交给子类去实现
        raise NotImplementedError       
        
    def prediction_layer(self,sequence_emb,target_emb):
        """
            input:
                sequence_emb_list: [B, D] 
                target_emb: [B, D]

            output:
                logit: [B, 1]
                
            功能：
                拼接sequence_emb与target_emb
        """
        final_prediction_embd = torch.cat([sequence_emb,target_emb], dim=-1)
        logit = self.final_deep_layers(final_prediction_embd)
        return logit
        
       
    def item_emb2logit(self,user_sequence_emb,target_item_emb):
        """
            input:
                user_sequence_emb: [B, L, D] 
                target_emb: [B, 1, D]

            output:
                logit_esu: [B, 1]
                logit_gsu: [B, 1]
                
            功能：
                拼接sequence_emb与target_emb
        """
        pos_relative_gsu_out_topk , pos_gsu_merged= self.stageone_softsearch(user_sequence_emb , target_item_emb)
        esu_output_pos, _ = self.stagetwo_embdfusion(pos_relative_gsu_out_topk , target_item_emb)
        # torch.squeeze可以干掉张量中所有维度为1的shape
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )
        logit_pos_gsu_merged = self.prediction_layer(pos_gsu_merged ,torch.squeeze(target_item_emb) )
        
        return logit_pos_esu , logit_pos_gsu_merged
              
    def get_logit_pos_neg(self,input_ids,target_item_id_pos,target_item_id_neg):
        """
            input:
                input_ids: [B, L], historical Id
                target_item_id_pos: [B, 1], last item of target positive item id sequence
                target_item_id_neg: [B, 1], last item of target negative item id sequence

                or
                target_item_id_neg: [B, num_negative], num_negative item of target negative item id sequence

            output:
                logits_pos:  [B, 1]
                logits_neg:  [B, 1]
                
                or
                logits_neg:  [B, num_negative]
            
            功能：
                将输入与user的itemid序列，以及正负item id得到正负样本的logit
        """
        # 1.得到embedding
        input_items_embd = self.item_embed_layer(input_ids)
        target_pos_item_embd = self.item_embed_layer(target_item_id_pos)
        target_neg_item_embd = self.item_embed_layer(target_item_id_neg)
        
        # 2.得到positive item的logit
        # todo 查证一下sim是不是这么训练的
        # todo 没准topk这个操作是可微的
        # todo 我想要得到一个无需拆分的，直接outputcandidate，到时候再说，现在先把最基础的完成了
        # todo 这个无拆分的可以得到说明一个优势，就是众candidate，可能可以优化计算，也未必吧，到时候再说
        pos_esu_logit , pos_gsu_logit = self.item_emb2logit(input_items_embd , target_pos_item_embd)
        
        # 3.得到negative item的logit
        logits_neg_list = []
        logits_neg_gsu_merged_list = []
        num_negative = target_item_id_neg.shape[-1]
        for i in range(num_negative):
            target_item_id_neg_singal_embd = target_neg_item_embd[:, i , :].unsqueeze(1)
            neg_esu_logit , neg_gsu_logit = self.item_emb2logit(input_items_embd , target_item_id_neg_singal_embd)

            logits_neg_list.append(neg_esu_logit)
            logits_neg_gsu_merged_list.append(neg_gsu_logit)
        
        logits_neg_group = torch.cat(logits_neg_list, dim=-1)
        logits_neg_gsu_merged_group = torch.cat(logits_neg_gsu_merged_list, dim=-1)
        # 4.返回logit,所以可以统一起来写
        return pos_esu_logit, logits_neg_group, pos_gsu_logit, logits_neg_gsu_merged_group
        
        