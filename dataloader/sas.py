from .base import AbstractDataloader

import torch
import random
import numpy as np
import torch.utils.data as data_utils


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class SASDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap) # 从1开始,1...item_count 为itme id

        args.num_users = self.user_count
        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.sliding_size = args.sliding_window_size
        # 我们应该没有这个mask token，还需要查明的是我们的itemid 是否从0开始,
        self.item_size = self.item_count + 1
        args.item_size = self.item_size

    @classmethod
    def code(cls):
        return 'sas'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=self.args.num_workers,
                        worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = SASTrainDataset(
            self.args, self.train, self.max_len, self.sliding_size, self.rng , self.item_count)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = SASValidDataset(self.args, self.train, self.val, self.max_len, self.rng,self.item_count)
        elif mode == 'test':
            dataset = SASTestDataset(self.args, self.train, self.val, self.test, self.max_len, self.rng,self.item_count)
        return dataset

# 添加负样本，动态添加，配合滑动窗口,正负样本1：1先把，这样快一些
# allseq存储的是各个滑动后的产生序列，不需要再额外添加了，应该是maxlen+1的长度
# 最后一位作为正样本，也就是target_pos_item，然后随机抽一位作为负样本（不在序列也不在正样本里），作为target_neg_item
class SASTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, sliding_size, rng,item_count):
        self.args = args
        self.max_len = max_len # 序列长度
        self.sliding_step = int(sliding_size * max_len)
        self.num_items = args.num_items
        self.rng = rng
        self.item_count = item_count
        
        assert self.sliding_step > 0
        self.all_seqs = []
        self.pos_item = []
        self.neg_item = [] # 直接从seq中无放回抽滑动次数个
        all_item_set = {i for i in range(1,self.item_count+1)} # 所有item的集合,【1-self.item_count】
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            neg_item_pool = all_item_set - set(seq)
            # 序列长度小则，直接加
            if len(seq) < self.max_len + self.sliding_step:
                # continue
                self.all_seqs.append(seq[:-1])
                self.pos_item.append(seq[-1])
                self.neg_item.append(random.sample(neg_item_pool,1)[0]) # 在除了seq的item中随机抽一个，也不要是0
            else:
                # 序列长度大于则拆开，相当于是滑动窗口版本了，start_idx就是起始idx，从最后一个能有max_LEN的到最后一位。
                start_idx = range(len(seq) - max_len -1, -1, -self.sliding_step)
                # append是加到最后一位，+是将【】解包，然后在逐个append。
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]
                self.pos_item = self.pos_item + [seq[i+max_len] for i in start_idx]
                # TODO 有没有可能len(start_idx)大于了neg_item_pool，应该不可能吧
                self.neg_item = self.neg_item + random.sample(neg_item_pool,len(start_idx))
                # print(f"{len(seq)} and {self.sliding_step} \n\n\n\n { self.all_seqs} \n\n\n\n {self.pos_item}  \n\n\n\n  {self.neg_item}  \n\n\n\n  {[i for i in start_idx]}")
                # break
                
    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        # index与user解绑了
        # 还需要加上0
        seq = self.all_seqs[index][-self.max_len:]
        mask_len = self.max_len - len(seq)
        seq = [0] * mask_len + seq
        target_pos = self.pos_item[index]
        target_neg = self.neg_item[index]
        # assert (set(seq + [target_pos]) & set([target_neg])) == set()
        # print(f"2 {seq} \n\n\n 3 {target_pos} \n\n\n\n 4 {target_neg}")
        # print(f"{len(seq)} {len(target_pos)} {len(target_neg)}")
        # finally find the reason
        return torch.LongTensor(seq), torch.tensor([target_pos], dtype = torch.long) , torch.tensor([target_neg], dtype = torch.long)
        
        # label比token多一个,多一个最后一个元素
        # labels = seq[-self.max_len:]
        # # [:-1]指的是除掉最后一个，[-self.max_len:]是最后
        # tokens = seq[:-1][-self.max_len:]
        # mask_len = self.max_len - len(tokens)
        # tokens = [0] * mask_len + tokens
        # mask_len = self.max_len - len(labels)
        # labels = [0] * mask_len + labels
        
         
        # return torch.LongTensor(tokens), torch.LongTensor(labels)


class SASValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, rng, item_count):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        # user列表
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng
        self.item_count = item_count
        all_item_set = {i for i in range(1,self.item_count+1)}
        self.neg_item = {}
        for u in self.users:
            neg_item_pool = all_item_set - set(self.u2seq[u]) - set(self.u2answer[u])
            self.neg_item[u] = random.sample(neg_item_pool,self.args.negative_sample_size)
    
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        # 倒数第二个item
        answer = self.u2answer[user]
        

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        # assert (set(seq+answer)  & set(self.neg_item[user])) == set()
# seq直接使用-self.max_len:,负样本是从seq整理另外抽
        return torch.LongTensor(seq), torch.tensor(answer,dtype=torch.long), torch.LongTensor(self.neg_item[user])
        
        # cur_idx, negs = 0, []
        # samples = self.rng.randint(1, self.args.num_items+1, size=5*self.args.negative_sample_size)
        # while len(negs) < self.args.negative_sample_size:
        #     item = samples[cur_idx]
        #     cur_idx += 1
        #     if item in seq or item in answer: continue
        #     else: negs.append(item)

        # candidates = answer + negs
        # labels = [1] * len(answer) + [0] * len(negs)

        # seq = seq[-self.max_len:]
        # padding_len = self.max_len - len(seq)
        # seq = [0] * padding_len + seq

        # return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class SASTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, max_len, rng , item_count):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2val[u]) > 0 and len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng
        self.item_count = item_count
        all_item_set = {i for i in range(1,self.item_count+1)}
        self.neg_item = {}
        for u in self.users:
            neg_item_pool = all_item_set - set(self.u2seq[u]) - set(self.u2answer[u]) -set(self.u2val[u])
            self.neg_item[u] = random.sample(neg_item_pool,self.args.negative_sample_size)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        # assert (set(seq+answer) & set(self.neg_item[user])) == set()
# seq是-self.max_len:,包含了val
        return torch.LongTensor(seq), torch.LongTensor(answer) , torch.LongTensor(self.neg_item[user])

        # cur_idx, negs = 0, []
        # samples = self.rng.randint(1, self.args.num_items+1, size=5*self.args.negative_sample_size)
        # while len(negs) < self.args.negative_sample_size:
        #     item = samples[cur_idx]
        #     cur_idx += 1
        #     if item in seq or item in answer: continue
        #     else: negs.append(item)
        
        # candidates = answer + negs
        # labels = [1] * len(answer) + [0] * len(negs)

        # seq = seq[-self.max_len:]
        # padding_len = self.max_len - len(seq)
        # seq = [0] * padding_len + seq

        # return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)