from .base import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class SDIM(BaseSearchBasedModel):
    def __init__(self, args):
        super().__init__(args)
        self.args.num_hashes = self.args.num_heads
        self.args.hash_bits = self.args.gsu_embd_hidden_size
        self.num_hashes = self.args.num_hashes
        self.random_rotations = torch.randn(self.args.hidden_size,self.args.num_hashes
                                             ,self.args.hash_bits,device=self.args.device,requires_grad = False)
        self.powers_of_two = nn.Parameter(torch.tensor([2.0 ** i for i in range(self.args.hash_bits)]), 
                                          requires_grad=False)
    
    def lsh_hash(self, vecs):
        """ 
            Input: vecs, with shape B x seq_len x d
            Output: hash_bucket, with shape B x seq_len x num_hashes
        """
        # print(f"1 -> {vecs} 2->{vecs.shape}")

        rotated_vecs = torch.einsum("bld,dht->blht", vecs, self.random_rotations) # B x seq_len x num_hashes x hash_bits
        hash_code = torch.relu(torch.sign(rotated_vecs))
        hash_bucket = torch.matmul(hash_code, self.powers_of_two.unsqueeze(-1)).squeeze(-1)
        return hash_bucket
    
    def lsh_attentioin(self, target_item, history_sequence):
        """ 
            input:
                target_item [B,D]
                history_sequence [B,L,D]
            output:
                attn_out [B,D]
        """
        # num_hashes表示的是分桶的数量，hash_bits表示的是一个通的bit数量
        # 我们要得到的是对分桶内做求和，分桶间做avgpooling
        # 分桶聚合的对象是L序列，也就是要将BLD-》BD
        target_bucket = self.lsh_hash(history_sequence)
        sequence_bucket = self.lsh_hash(target_item.unsqueeze(1))
        # 之所以转置是因为BLD，计算碰撞数量，collide_mask表示的序列中标识了发生碰撞的位置
        bucket_match = (sequence_bucket - target_bucket).permute(2, 0, 1) # num_hashes x B x seq_len
        collide_mask = (bucket_match == 0).float()
        # torch.nonzero将为1的位置提取出来，得到collide_index，collide_index表示的是序列中发生碰撞的index，
        # 且重复了num_hashes次
        # print(f"1->{collide_mask}")
        hash_index, collide_index = torch.nonzero(collide_mask.flatten(start_dim=1), as_tuple=True)
        # .sum表示的是每一个batch每一个hash分桶中user序列与target_item碰撞的数量，
        # .cumsum表示的是前缀和，表示的是第i个batch的位置，offsets表示分袋的数量，且指示的是索引位置
        # print(f"2->{collide_index}")
        offsets = collide_mask.sum(dim=-1).long().flatten().cumsum(dim=0)
        offsets = offsets - offsets[0]
        # print(f"3->{collide_mask.sum(dim=-1)} 4->{offsets}")
        # 根据collide_index以及offsets，从history_sequence拼接向量并求和
        attn_out = F.embedding_bag(collide_index, history_sequence.view(-1, target_item.size(1)), 
                                   offsets, mode='sum') # (num_hashes x B) x d
        attn_out = attn_out.view(self.num_hashes, -1, target_item.size(1)).mean(dim=0) # B x d
        return attn_out
    
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
        # [B,L,D] ; [B,1,D]
        history_sequence , _ , target_item_emb = self.item_embed_layer(user_sequence_id,target_item_id)
        # print(f"3->{target_item_emb.shape} 4->{torch.squeeze(target_item_emb).shape}")
        esu_output_pos = self.lsh_attentioin(torch.squeeze(target_item_emb),history_sequence)
        logit_pos_esu = self.prediction_layer(esu_output_pos ,torch.squeeze(target_item_emb) )

        return logit_pos_esu , None