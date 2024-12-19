import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class BaseSearchBasedModel(nn.Module):
    """
        Baseline Model
    """
    def __init__(self, args):
        super().__init__()
        # 默认是esu、gsu、以及target item
        # [0,args.item_size-1] \ 并指定索引0为全零且不更新 \ 并且添加倒最后一维
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.final_deep_layers = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
        )
        
        self.wQ_list = [nn.Linear(args.hidden_size, args.hidden_size).to(self.args.device) for _ in range(self.args.num_heads)]
        self.wK_list = [nn.Linear(args.hidden_size, args.hidden_size).to(self.args.device) for _ in range(self.args.num_heads)]
        self.wV_list = [nn.Linear(args.hidden_size, args.hidden_size).to(self.args.device) for _ in range(self.args.num_heads)]
        self.wO = nn.Linear(self.args.num_heads * args.hidden_size, args.hidden_size)
        
        
        
    
    # 对用户行为序列、item 得到嵌入、
    def item_embed_layer(self,input_sequence_ids,target_item_id):
        """
            input:
                input_sequence_ids: shape: [B, L] type: int 
                target_item_id: shape: [B, 1] type: int 

            output:
                gsu_embd = [B, L, D]
                esu_embd = [B, L, D]
                target_item_embd = [B, 1, D]
                
            功能：
                物品嵌入层，将L个item的id转为embedding,
        """
        gsu_embd = self.item_embeddings(input_sequence_ids)
        esu_embd = gsu_embd
        target_item_embd = self.item_embeddings(target_item_id)
        return gsu_embd , esu_embd , target_item_embd

        
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
                利用target_emb在user_seq_emb search出最接近的K个item，但是是一个candidate_size,后续的性能测试/改进可以基于此
        """
        # 交给子类去实现
        raise NotImplementedError
        
    def stagetwo_embdfusion(self,indices,user_seq_emb ,target_emb):
        """
            input:
                indices: [B, K]
                user_seq_emb: [B, L, D] 
                target_emb: [B, 1, D]

            output:
                mhta_output: [B, D]
                attention_mask_list: list of attention_mask [B, L]
                
            功能：
                将sequence_emb融合为一个定长的embedding
        """
        # gather_index [B,K,D]
        emb_dim = user_seq_emb.shape[-1]
        gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64) 
        # user_seq_emb -> gsu_out_topk: [B, L, D] -> [B, K, D]
        # gsu_out_topk[i][j][k] = user_seq_emb_mask[i][gather_index[i][j][k]][k]
        gsu_out_topk = torch.gather(user_seq_emb, dim=1, index=gather_index, out=None)
        
        num_heads = self.args.num_heads
        head_list = []

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
    
    def stagetwo_allinone_embdfusion(self,user_seq_emb,target_emb):
        """
            input:
                sequence_emb_list: [B, candidate_size , K, D] 
                target_emb: [B , candidate_size, D]

            output:
                mhta_output: [B, candidate_size ,D]
                attention_mask_list: list of attention_mask [B, candidate_size, L]
                
            功能：
                将sequence_emb融合为一个定长的embedding,但是是一个candidate_size,后续的性能测试/改进可以基于此
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
        gsu_seq_embd , esu_seq_embd , target_item_emb = self.item_embed_layer(user_sequence_id,target_item_id)
        
        pos_relative_gsu_out_topk , pos_gsu_merged= self.stageone_softsearch(gsu_seq_embd , target_item_emb)
        esu_output_pos, _ = self.stagetwo_embdfusion(pos_relative_gsu_out_topk ,esu_seq_embd ,  target_item_emb)
        # torch.squeeze可以干掉张量中所有维度为1的shape
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )
        # logit_pos_gsu_merged = self.prediction_layer(pos_gsu_merged ,torch.squeeze(target_item_emb) )
        
        return logit_pos_esu , None
              
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
                将输入与user的itemid序列，以及正负item id得到正负样本的logit,这里要做的事情就粗糙一些，方便堕胎
        """
        
        
        # 2.得到positive item的logit
        # todo 查证一下sim是不是这么训练的
        # todo 没准topk这个操作是可微的
        # todo 我想要得到一个无需拆分的，直接outputcandidate，到时候再说，现在先把最基础的完成了
        # todo 这个无拆分的可以得到说明一个优势，就是众candidate，可能可以优化计算，也未必吧，到时候再说
        # TODO 具体实现的时候需要区分哪个模型要计算GSU loss
        pos_esu_logit , pos_gsu_logit = self.item_id2logit(input_ids , target_item_id_pos)
        
        # 3.得到negative item的logit
        logits_neg_list = []
        logits_neg_gsu_merged_list = []
        num_negative = target_item_id_neg.shape[-1]
        for i in range(num_negative):
            neg_esu_logit , neg_gsu_logit = self.item_id2logit(input_ids , target_item_id_neg[:,i].unsqueeze(1))

            logits_neg_list.append(neg_esu_logit)
            logits_neg_gsu_merged_list.append(neg_gsu_logit)
        
        logits_neg_group = torch.cat(logits_neg_list, dim=-1)
        try:
            logits_neg_gsu_merged_group = torch.cat(logits_neg_gsu_merged_list, dim=-1)
        except TypeError:
            logits_neg_gsu_merged_group = None
        # 4.返回logit,所以可以统一起来写
        return pos_esu_logit, logits_neg_group, pos_gsu_logit, logits_neg_gsu_merged_group
    
    
        
        
        