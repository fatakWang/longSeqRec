import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'
# 数据集代码
    @classmethod
    @abstractmethod
    def code(cls):
        pass
# cls代表的类本身,感觉写的有毛病
    @classmethod    
    def raw_code(cls):
        return cls.code()
# 
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset
    
    def remove_immediate_repeats(self, df):
        df_next = df.shift()
        is_not_repeat = (df['uid'] != df_next['uid']) | (df['sid'] != df_next['sid'])
        df = df[is_not_repeat]
        return df
# 过滤掉不满足要求的三元组
    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            # 按照sid groupby，得到sid交互数量
            item_sizes = df.groupby('sid').size()
            # index即为item_sizes索引，也就是item id，筛掉那些小于self.min_sc的itemid
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            # 过滤掉那些sid不满足要求的
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df
    
    def densify_index(self, df):
        print('Densifying index')
        # 
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(
                lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError
# data/
    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)
# data/ml-1m
    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())
# data/preprocessed
    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')
# data/data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-leave_one_out
    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)
# data/data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-leave_one_out/dataset.pkl (应该是一个文件，前面的都是目录),可以是文件也可以是文件夹
    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
