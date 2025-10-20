import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
from torch.utils.data import ConcatDataset
import logging
import re
import pdb
import json
from prompt import sft_prompt, all_prompt
import numpy as np
from transformers import T5Tokenizer

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class SasRecDataset(Dataset):
    def __init__(self, args, mode="train", neg_sample_num=1):
        super().__init__()
        self.mode = mode
        self.neg_sample_num = neg_sample_num
        self.args = args
        self.dataset = args.dataset
        
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len

        self._load_data()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        # check if user item are continues
        item_set = set()
        user_set = set()

        for k,v in  self.inters.items():
            item_set.update(set(v))
            user_set.add(int(k))
        self.item_set = item_set
        self.user_set = user_set
        self.user_num = len(user_set)
        self.item_num = len(item_set)
        max_item,min_item,max_user,min_user = max(item_set),min(item_set),max(user_set),min(user_set)
        assert self.item_num == max_item-min_item+1 and self.user_num == max_user-min_user+1 and min_item==0 and min_user==0
        self.user_D,self.item_D = np.zeros(self.user_num),np.zeros(self.item_num)
        for k,v in  self.inters.items():
            item_list = v[:-2] # traindata
            self.user_D[int(k)] += len(item_list)
            self.item_D[item_list] += 1
    
    def construct_neg_item_list(self,ts):
        neg_item_list = []
        if self.neg_sample_num<0:
            neg_item_list.append(-1)
        else:
            for i in range(self.neg_sample_num):
                neg_item_list.append(random_neq(0,self.item_num,ts))
        return neg_item_list


    def _process_train_data(self):
        inter_data = [] # 由one_data组成，
        for uid  in self.inters:
            items = self.inters[uid][:-2]
            ts = set(self.inters[uid])
            for i in range(1, len(items)):
                # minnum element item and inters
                # pos_item : '6' (B,1) 
                # neg_item: (B,K)
                # mask: (B,L)
                # input_ids : np. (B,L)
                one_data = dict() # 字如其名
                one_data["user"] = torch.tensor([int(uid)])
                one_data["pos_item"] = torch.tensor([items[i]])
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data["input_ids"] = torch.ones(self.max_his_len) * self.item_num
                one_data["input_ids"][-len(history):] = torch.tensor(history)
                one_data["attention_mask"] = torch.zeros(self.max_his_len,dtype=torch.bool)
                one_data["attention_mask"][-len(history):] = True
                # from all item sample self.neg_sample_num 
                neg_item_list = self.construct_neg_item_list(ts)
                
                one_data["neg_item"] = torch.tensor(neg_item_list)
                inter_data.append(one_data)
            
            if self.args.debug and int(uid) >100:
                break

        return inter_data
    
    def _process_valid_data(self):
# validation for the early stop
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            ts = set(self.inters[uid])
            one_data = dict()
            one_data["user"] = torch.tensor([int(uid)])
            one_data["pos_item"] = torch.tensor([items[-2]])
            history = items[:-2] # list
            one_data["input_ids"] = torch.ones(self.max_his_len) * self.item_num
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["input_ids"][-len(history):] = torch.tensor(history)
            one_data["attention_mask"] = torch.zeros(self.max_his_len,dtype=torch.bool)
            one_data["attention_mask"][-len(history):] = True
            neg_item_list = self.construct_neg_item_list(ts)

            one_data["neg_item"] = torch.tensor(neg_item_list)
            inter_data.append(one_data)

        return inter_data
    
    # TODO 研究一下tiger的data colactor 如何
    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            ts = set(self.inters[uid])
            one_data = dict()
            one_data["user"] = torch.tensor([int(uid)])
            one_data["pos_item"] = torch.tensor([items[-1]])
            history = items[:-1] # list
            one_data["input_ids"] = torch.ones(self.max_his_len) * self.item_num
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["input_ids"][-len(history):] = torch.tensor(history)
            one_data["attention_mask"] = torch.zeros(self.max_his_len,dtype=torch.bool)
            one_data["attention_mask"][-len(history):] = True
            neg_item_list = self.construct_neg_item_list(ts)

            one_data["neg_item"] = torch.tensor(neg_item_list)
            inter_data.append(one_data)

        return inter_data
    
    def __len__(self):
        return len(self.inter_data)
    
    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(user=d["user"].int(),input_ids=d["input_ids"].int(), pos_item=d["pos_item"].int(),neg_item=d["neg_item"].int(),attention_mask=d["attention_mask"])

class LightGCNDataset(Dataset):
    def __init__(self, args, mode="train", neg_sample_num=1):
        super().__init__()
        self.mode = mode
        self.args = args
        self.neg_sample_num = neg_sample_num
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self._load_data()
        

        if self.mode == 'train':
            self.inter_data = self._process_train_data()
            self.get_graph()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        # check if user item are continues
        item_set = set()
        user_set = set()

        for k,v in  self.inters.items():
            item_set.update(set(v))
            user_set.add(int(k))
        self.item_set = item_set
        self.user_set = user_set
        self.user_num = len(user_set)
        self.item_num = len(item_set)
        max_item,min_item,max_user,min_user = max(item_set),min(item_set),max(user_set),min(user_set)
        assert self.item_num == max_item-min_item+1 and self.user_num == max_user-min_user+1 and min_item==0 and min_user==0
        self.user_D,self.item_D = np.zeros(self.user_num),np.zeros(self.item_num)
        for k,v in  self.inters.items():
            item_list = v[:-2] # traindata
            self.user_D[int(k)] += len(item_list)
            self.item_D[item_list] += 1
    
    def get_graph(self):
        def v1():
            # user_tensor = torch.tensor([data["user"] for data in self.inter_data])
            # item_tensor = torch.tensor([data["pos_item"] for data in self.inter_data])
            # item_tensor += self.user_num

            # row_tensor = torch.concat((user_tensor,item_tensor)).long()
            # col_tensor = torch.concat((item_tensor,user_tensor)).long()
            # indice = torch.stack((row_tensor,col_tensor),dim=0)
            # coo_tensor = torch.sparse_coo_tensor(indice, torch.ones_like(indice[0],dtype=torch.float), 
            #                             (self.user_num+self.item_num,self.user_num+self.item_num))
            # rowsum = coo_tensor.sum(dim=1).float().to_dense()
            # d_inv = rowsum ** -0.5
            # d_inv[torch.isinf(d_inv)] = 0
            # d_inv = d_inv.diag().to_sparse_coo()
            # d_inv

            # A = d_inv @coo_tensor @ d_inv
            # self.A = A
            pass
        
        user_tensor = torch.tensor([data["user"] for data in self.inter_data])
        item_tensor = torch.tensor([data["pos_item"] for data in self.inter_data])
        item_tensor += self.user_num
        D = torch.concat((torch.tensor(self.user_D),torch.tensor(self.item_D))).float()
        D = D**-0.5
        D[torch.isinf(D)] = 0
        row_tensor = torch.concat((user_tensor,item_tensor)).long()
        col_tensor = torch.concat((item_tensor,user_tensor)).long()
        indice = torch.stack((row_tensor,col_tensor),dim=0)
        value_tensor = D[row_tensor] * D[col_tensor]
        self.A = torch.sparse_coo_tensor(indice, value_tensor, 
                                    (self.user_num+self.item_num,self.user_num+self.item_num))
        
    def construct_neg_item_list(self,ts):
        neg_item_list = []
        if self.neg_sample_num<0:
            neg_item_list.append(-1)
        else:
            for i in range(self.neg_sample_num):
                neg_item_list.append(random_neq(0,self.item_num,ts))
        return neg_item_list
    
    def _process_train_data(self):
        inter_data = []
        for uid  in self.inters:
            items = self.inters[uid][:-2]
            ts = set(self.inters[uid])
            for i in range(0, len(items)):
                one_data = dict()
                one_data["pos_item"] = torch.tensor([items[i]])
                one_data["user"] = torch.tensor([int(uid)])
                neg_item_list = self.construct_neg_item_list(ts)
                one_data["neg_item"] = torch.tensor(neg_item_list)
                inter_data.append(one_data)
            
            if self.args.debug and int(uid) >100:
                break
        return inter_data
    
    def _process_valid_data(self):
        inter_data = []
        for uid  in self.inters:
            one_data = dict()
            ts = set(self.inters[uid])
            one_data["pos_item"] = torch.tensor([self.inters[uid][-2]])
            one_data["user"] = torch.tensor([int(uid)])
            neg_item_list = self.construct_neg_item_list(ts)
            one_data["neg_item"] = torch.tensor(neg_item_list)
            inter_data.append(one_data)
            
        return inter_data
    
    def _process_test_data(self):
        inter_data = []
        for uid  in self.inters:
            one_data = dict()
            ts = set(self.inters[uid])
            one_data["pos_item"] = torch.tensor([self.inters[uid][-1]])
            one_data["user"] = torch.tensor([int(uid)])
            neg_item_list = self.construct_neg_item_list(ts)
            one_data["neg_item"] = torch.tensor(neg_item_list)
            inter_data.append(one_data)
            
        return inter_data
    
    def __len__(self):
        return len(self.inter_data)
    
    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(user=d["user"].int(), pos_item=d["pos_item"].int(),neg_item=d["neg_item"].int())
    
class SemanticsTokenDataset(Dataset):
    def __init__(self, args, mode="train",**kwargs):
        self.mode = mode
        # tiger no use negative item, softmax loss
        # self.neg_sample_num = neg_sample_num
        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.all_items = None
        self.max_his_len = args.max_his_len
        self.new_tokens = None

        self._load_data()
        self._remap_items()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".index.json"), 'r') as f:
            self.indices = json.load(f)

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def _remap_items(self):
        """
        "0":
        ['<a_183><b_70><c_232><d_6>', # 代表一个item
        '<a_183><b_122><c_96><d_235>',
        '<a_52><b_170><c_54><d_27>',
        '<a_110><b_228><c_51><d_223>',
        '<a_9><b_135><c_195><d_111>']
        """
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            # "".join 意思就是中间没间隔
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items # 

    def _process_train_data(self):
        inter_data = [] # 由one_data组成，
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                # minnum element item and inters
                # item : <a_9><b_135><c_195><d_111>
                # inters : '<a_183><b_70><c_232><d_6><a_183><b_122><c_96><d_235><a_52><b_170><c_54><d_27><a_110><b_228><c_51><d_223>'
                one_data = dict() # 字如其名
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]

                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):
        # validation for the early stop
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2] # list
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data
    
    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)
        return inter_data
    
    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])

# multi token dataset
class MultiTokenDataset(SemanticsTokenDataset):
    def __init__(self, args, tokenizer, mode='train',**kwargs):
        super().__init__(args, tokenizer, mode)


class SemanticsTokenCollator(object):
    def __init__(self, args, tokenizer,**kwargs):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):
        
 # labels : <a_9><b_135><c_195><d_111>
 # input_ids : '<a_183><b_70><c_232><d_6><a_183><b_122><c_96><d_235><a_52><b_170><c_54><d_27><a_110><b_228><c_51><d_223>'

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest", # # 填充至批次内最长序列长度
                                max_length=self.tokenizer.model_max_length, 
                                truncation=True, # 超过max_length则截断
                                return_attention_mask=True) # 返回注意力掩码

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100 # 这里的labels一般都是只有一个物品的，所以不会发生这种情况，T5如何处理pad token id

        """
        {
                'input_ids': tensor([[101, 202, 303, 404,   0,   0],
                                    [105, 206, 307, 408, 109,   0]]), # (B,L)
                'attention_mask': tensor([[1, 1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1, 0]]), # (B,L)
                'labels':

            }
        """

        return inputs

class LcrecCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # print(self.tokenizer.model_max_length)
        

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels


        return inputs
    
class LcrecTestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            padding_side="left",
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        inputs["labels"] = self.tokenizer(targets,
                                          return_tensors="pt",
                                          padding="longest",
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=True,
                                          return_attention_mask=True)["input_ids"]
        return inputs


class BaseDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = tokenizer("Response:")["input_ids"][1:]
# 因为decoder only模型生成的是一长串，只有当Response:之后的字符才需要限制生成。
        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError

class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        if "sample_num" not in args.__dict__:
            args.sample_num = -1
        self.sample_num = args.sample_num

        self.prompts = all_prompt["seqrec"]


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
                    
    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        # print(index, idx)

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        # print({"input": input, "output": output})

        return dict(input_ids=input, labels=output)

class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["fusionseqrec"]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = "".join(self.indices[str(items[i])])
                one_data["title"] = self.item_feat[str(items[i])]["title"].strip().strip(".!?,;:`")
                one_data["description"] = self.item_feat[str(items[i])]["description"]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                inters = ["".join(self.indices[str(j)]) for j in history]
                inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]


                if self.add_prefix:
                    inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                    inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

                one_data["inters"] = self.his_sep.join(inters)
                one_data["inter_titles"] = self.his_sep.join(inter_titles)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-2])])
            one_data["title"] = self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-2])]["description"]


            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-1])])
            one_data["title"] = self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-1])]["description"]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)


        return dict(input_ids=input, labels=output)

class ItemFeatDataset(BaseDataset):

    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        # load data
        self._load_data()
        self.feat_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)


    def _process_data(self):

        feat_data = []
        for iid in self.item_feat:
            feat = self.item_feat[iid]
            index = "".join(self.indices[iid])
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")
            feat_data.append(feat)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data


    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)

class ItemSearchDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["itemsearch"]

        # load data
        self._load_data()
        self.search_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)


    def _process_data(self):

        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        if self.mode == 'train':
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == 'test':
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError

        for uid in user_explicit_preference.keys():
            one_data = {}
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep
            one_data["user_related_intention"] = user_vi[0]
            one_data["item_related_intention"] = user_vi[1]

            iid = user_vague_intention[uid]["item"]
            inters = user_vague_intention[uid]["inters"]

            index = "".join(self.indices[str(iid)])
            one_data["item"] = index

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            inters = ["".join(self.indices[str(i)]) for i in inters]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["inters"] = self.his_sep.join(inters)

            search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.search_data) * self.prompt_sample_num
        elif self.mode == 'test':
            return len(self.search_data)
        else:
            return len(self.search_data)


    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.search_data[idx]
        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]
# one_data 有 explicit_preferences（长度为3的列表），user_related_intention item_related_intention  item  inters ，还有加入了query。
        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)

class PreferenceObtainDataset(BaseDataset):

    def __init__(self, args, prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        # load data
        self._load_data()
        self._remap_items()

        self.preference_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):

        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference.keys():
            one_data = {}
            inters = self.remapped_inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
# explicit_preferences 有三个 preference，后面要随机抽样一个出来
            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.preference_data) * self.prompt_sample_num


    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)

class lcrecDataset(object):
    def __init__(self, args, mode ,**kwargs):
        assert args.model_code in ["lcrec"]
        self.mode = mode
        if mode == "train":
            tasks = args.tasks.split(",")
            train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
            assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
            train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
            assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

            train_datasets = []
            for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
                if task.lower() == "seqrec":
                    dataset = SeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
                elif task.lower() == "item2index" or task.lower() == "index2item":
                    dataset = ItemFeatDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
                elif task.lower() == "fusionseqrec":
                    dataset = FusionSeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
                elif task.lower() == "itemsearch":
                    dataset = ItemSearchDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
                elif task.lower() == "preferenceobtain":
                    dataset = PreferenceObtainDataset(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
                else:
                    raise NotImplementedError
                train_datasets.append(dataset)
            self.dataset_ = ConcatDataset(train_datasets)
        elif mode=="valid":
            self.dataset_ = SeqRecDataset(args,"valid",prompt_sample_num = args.valid_prompt_sample_num)
        elif mode=="test":
            self.dataset_ = SeqRecDataset(args,"test")

    def get_all_items(self):
        if self.mode=="train":
            return self.dataset_.datasets[0].get_all_items()
        else:
            return self.dataset_.get_all_items()
    
    def get_new_tokens(self):
        if self.mode == "train":
            return self.dataset_.datasets[0].get_new_tokens()
        else:
            return self.dataset_.get_new_tokens()

    def __len__(self):
        return len(self.dataset_)
    
    def __getitem__(self, index):
        return self.dataset_[index]