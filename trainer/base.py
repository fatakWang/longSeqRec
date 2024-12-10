from model import *
from config import *
from .utils import *
from .loggers import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import json
import numpy as np
from abc import ABCMeta
from pathlib import Path
from collections import OrderedDict


class BaseTrainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb=True):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(self.train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)
            
        self.export_root = export_root
        if not os.path.exists(self.export_root):
            Path(self.export_root).mkdir(parents=True)
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(
                name=self.args.model_code+'_'+self.args.dataset_code,
                project=PROJECT_NAME,
                config=args,
            )
            writer = wandb
        else:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(
                log_dir=Path(self.export_root).joinpath('logs'),
                comment=self.args.model_code+'_'+self.args.dataset_code,
            )
        self.val_loggers, self.test_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.args, writer, self.val_loggers, self.test_loggers, use_wandb)
        
        print(args)
        print('Total parameters:', sum(p.numel() for p in model.parameters()))
        print('Encoder parameters:', sum(p.numel() for n, p in model.named_parameters() \
                                         if 'embedding' not in n))
# 调用train_one_epoch得到每一轮的训练指标，并且定期调用vaild监测何时停下来
    def train(self):
        accum_iter = 0
        self.exit_training = self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            if self.args.val_strategy == 'epoch':
                self.exit_training = self.validate(epoch, accum_iter)  # val after every epoch
            if self.exit_training:
                print('Early stopping triggered. Exit training')
                break
        self.logger_service.complete()
# 计算loss，并且把这一epoch每一个iter的loss平均值记录在 tqdm上，并且定期调用vaild监测是否停止训练
    def train_one_epoch(self, epoch, accum_iter):
        average_meter_set = AverageMeterSet()
        # 一个epoch记录一次
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.model.train()
            batch = self.to_device(batch)

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            self.clip_gradients(self.args.max_grad_norm)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            average_meter_set.update('loss', loss.item())
            # 实现一个提示信息
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += 1
            if self.args.val_strategy == 'iteration' and accum_iter % self.args.val_iterations == 0:
                self.exit_training = self.validate(epoch, accum_iter)  # val after certain iterations
                if self.exit_training: break


        return accum_iter
# 激素那每一个iter的指标，记录在tqdm以及logger_SERVER里
    def validate(self, epoch, accum_iter):
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
        
        return self.logger_service.log_val(log_data)  # early stopping
# 使用最好的vaild模型计算指标，每一个iter得到一次指标，并将全局的指标更新在logger_service，并将全局指标打印出来
    def test(self, epoch=-1, accum_iter=-1):
        print('******************** Testing Best Model ********************')
        best_model_dict = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        # 与vaild的区别就是load了best_model_dict
        self.model.load_state_dict(best_model_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                metrics = self.calculate_metrics(batch)
                # 将metrics更新进average_meter_set,metrics是一个kv,k是key，
                # v是value，key是存在average_meter_set的，v是对应一个iter的对应指标平均值
                # 直接更新就好了，是等价的
                self._update_meter_set(average_meter_set, metrics)
                # 更新dataloader_metrics的描述
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            average_metrics = average_meter_set.averages()
            log_data.update(average_metrics)
            self.logger_service.log_test(log_data)
# 最终的产出就是 average_metrics name:value
            print('******************** Testing Metrics ********************')
            print(average_metrics)
            with open(os.path.join(self.export_root, 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics
    
    def to_device(self, batch):
        return [x.to(self.device) for x in batch]

    def cross_entropy_next_item_loss(self, logits_pos, logits_neg_batch):
        """
            logits_pos: [B, 1]
            logits_neg_group: [B, nun_sample]
            nn.BCELoss() inputs: pred after sigmoid, label 0/1
        """
        ## Some Long Sequence Model, First Stage Embedding Doesn't Participate In the Final Loss
        if logits_pos is None or logits_neg_batch is None:
            return 0.0
        ## [batch_size, num_sample] -> [batch_size * num_sample, 1]
        logits_neg_batch_reshape = torch.reshape(logits_neg_batch, shape=[-1]).unsqueeze(-1)
        logits_merge = torch.cat([logits_pos, logits_neg_batch_reshape], dim=0)
        label_merge = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg_batch_reshape)], dim=0)
        pred_merge = nn.Sigmoid()(logits_merge)
        loss = nn.BCELoss()(pred_merge, label_merge) # 返回的是1
        return loss
    
    def calculate_loss(self, batch):
        # [B,L] , [B,1] ,[B,1]
        seqs, labels ,target_neg  = batch
        pos_esu_logit, logits_neg_group, pos_gsu_logit, logits_neg_gsu_merged_group = self.model.get_logit_pos_neg(seqs,labels ,target_neg)
        if self.args.no_gsu_loss
            loss_gsu = self.cross_entropy_next_item_loss(logits_stage1_pos, logits_stage1_neg)
        else:
            loss_gsu = 0
        loss_esu = self.cross_entropy_next_item_loss(logits_pos, logits_neg_group)
        loss = loss_gsu + loss_esu
        
        return loss
    
    def get_sample_scores(self, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        Recall_1, NDCG_1 = get_metric(pred_list, 1)
        Recall_5, NDCG_5= get_metric(pred_list, 5)
        Recall_10, NDCG_10 = get_metric(pred_list, 10)
        Recall_50, NDCG_50 = get_metric(pred_list, 50)

        post_fix = {
            "Recall@1": '{:.4f}'.format(Recall_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "Recall@5": '{:.4f}'.format(Recall_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "Recall@10": '{:.4f}'.format(Recall_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "Recall@50": '{:.4f}'.format(Recall_50), "NDCG@50": '{:.4f}'.format(NDCG_50)
        }
       
        return post_fix
    
    
    def calculate_metrics(self, batch):
        # [B,L] , [B,1] ,[B,100]
        seqs, labels ,target_neg  = batch
        # TODO model gsu返回的是none需要改造，num_sample需要确认
        pos_esu_logit, logits_neg_group, pos_gsu_logit, logits_neg_gsu_merged_group = self.model.get_logit_pos_neg(seqs,labels ,target_neg)
        test_logit = torch.cat([pos_esu_logit, logits_neg_group], dim=-1).cpu().detach().numpy().copy()

        return self.get_sample_scores(test_logit)
        
        
        
    def clip_gradients(self, limit=1.0):
        nn.utils.clip_grad_norm_(self.model.parameters(), limit)
# 挨个upadate
    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        # ['NDCG@1', 'NDCG@5', 'NDCG@10', 'Recall@1', 'Recall@5', 'Recall@10']
        description = 'Eval: ' + \
            ', '.join(s + ' {:.4f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        # 'Eval: N@1 {:.4f}, N@5 {:.4f}, N@10 {:.4f}, R@1 {:.4f}, R@5 {:.4f}, R@10 {:.4f}'
        # {}可以接受format , 
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)
# val_LOGGER 有三个MetricGraphPrinter ， 和 RecentModelLogger  BestModelLogger
    def _create_loggers(self):
        root = Path(self.export_root)
        model_checkpoint = root.joinpath('models')

        val_loggers, test_loggers = [], []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation', use_wandb=self.use_wandb))
            val_loggers.append(
                MetricGraphPrinter(key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation', use_wandb=self.use_wandb))

        val_loggers.append(RecentModelLogger(self.args, model_checkpoint))
        val_loggers.append(BestModelLogger(self.args, model_checkpoint, metric_key=self.best_metric))

        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Test', use_wandb=self.use_wandb))
            test_loggers.append(
                MetricGraphPrinter(key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Test', use_wandb=self.use_wandb))

        return val_loggers, test_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }