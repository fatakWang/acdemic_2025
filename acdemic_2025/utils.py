import math
import os
import random
import datetime
from typing import Dict, List
import transformers
import yaml
import numpy as np
import torch
import torch.nn as nn
from data import *
import baseline_config
import baseline_model
import argparse
from transformers.models.t5.configuration_t5 import T5Config
import optuna
import copy
import wandb
import gc
# decide not to use api ,because it's not flexible
PURPLE = '\033[35m'
BOLD = '\033[1m'
RESET = '\033[0m'

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

class DynamicClass:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)

class Trie(object):
    # 匹配到了就返回对应的list(keys)，没匹配到就是[]，或者全匹配到了就是[]
    # {1：{2：{3：{}}}}
    #  
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return [] # 啥都没的时候就return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
    args = DynamicClass(config)
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

dataset_dict = {
        "sasrec": SasRecDataset,
        "lightgcn": LightGCNDataset,
        "mf": LightGCNDataset,
        "caser": SasRecDataset,
        "hgn": SasRecDataset,
        "tiger": SemanticsTokenDataset,
        "lcrec": lcrecDataset,
    }

def config_factory(args):
    config_dict = {
        "sasrec":baseline_config.SASREConfig,
        "lightgcn": baseline_config.LightGCNConfig,
        "mf" : baseline_config.mfConifg,
        "caser": baseline_config.caserConfig,
        "hgn": baseline_config.hgnConfig,
        "tiger": T5Config,
    }
    return config_dict[args.model_code](**args.__dict__)

def model_factory(args):
    model_dict = {
        "sasrec":baseline_model.SASREC,
        "lightgcn": baseline_model.LightGCN,
        "mf": baseline_model.mf,
        "caser": baseline_model.caser,
        "hgn": baseline_model.hgn,
        "tiger": baseline_model.TIGER,
        "lcrec": baseline_model.LCREC,
    }
    if args.model_code in ["lcrec"]:
        
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        if ddp:
            device_map = {"": local_rank}
        else:
            device_map = "auto"
        return model_dict[args.model_code].from_pretrained(args.base_model,
                    load_in_4bit=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,)
    else:
        return model_dict[args.model_code](config_factory(args))

def load_datasets(args):
    train_data = dataset_dict[args.model_code](args=args,mode="train",neg_sample_num=args.train_data_sample_num)
    valid_data = dataset_dict[args.model_code](args=args,mode="valid",neg_sample_num=args.test_sample_num)
    return train_data, valid_data

def creat_tokenizer(args):
    tokenzier_dict = {
        "tiger": transformers.AutoTokenizer,
        "lcrec": transformers.AutoTokenizer,
    }
    if args.model_code in ["tiger"]:
        return tokenzier_dict[args.model_code].from_pretrained(
            args.base_model,
            model_max_length=512,
        )
    elif args.model_code in ["lcrec"]:
        return tokenzier_dict[args.model_code].from_pretrained(
            args.base_model,
            model_max_length=512,
        )
    else:
        return None

collator_dict = {
    "sasrec": None,
    "lightgcn": None,
    "mf": None,
    "caser": None,
    "hgn": None,
    "tiger": SemanticsTokenCollator,
    "lcrec": LcrecCollator,
}

def creat_collator(args,tokenizer):
    if collator_dict[args.model_code] is not None:
        return collator_dict[args.model_code](args,tokenizer)
    else:
        return None

def creat_trainer(args,model,train_data,valid_data,tokenizer):
    compute_metrics_object = StatefulComputeMetrics()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics_object if args.model_code  not in ["tiger","lcrec"] else None,
        args=transformers.TrainingArguments( # here
            seed=args.seed,
            run_name=args.task_name,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio, # 0.01
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay, # 0.01
            lr_scheduler_type=args.lr_scheduler_type, # cosine
            logging_steps=args.logging_steps, # 10
            optim=args.optim, # adamw
            eval_strategy=args.eval_save_strategy, # epoch
            save_strategy=args.eval_save_strategy,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit, # 限制保存的检查点（checkpoint）总数
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            label_names=args.label_names,
            greater_is_better=args.greater_is_better,
            batch_eval_metrics=True,
            eval_delay= 1 if args.eval_save_strategy=="epoch" else 2000,
            prediction_loss_only=False if args.model_code not in ["tiger","lcrec"] else True,
            fp16=args.fp16 if args.model_code  in ["lcrec"] else None,
            bf16=args.bf16 if args.model_code  in ["lcrec"] else None, 
            deepspeed=args.deepspeed if args.model_code  in ["lcrec"] else None,
            ddp_find_unused_parameters=False if ddp else None,
            resume_from_checkpoint=args.resume_from_checkpoint,
            no_cuda=args.no_cuda,
        ),
        tokenizer=tokenizer,
        data_collator=creat_collator(args,tokenizer),
        callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if (args.eval_save_strategy != 'no') else None,
    )
    return trainer

def load_test_dataset(args):
    test_data = dataset_dict[args.model_code](args=args,mode="test",neg_sample_num=args.test_sample_num)

    return test_data

def compute_bce_loss(logit, pos_item, neg_item):
    """
    计算二分类任务的BCE Loss
    Args:
        logit: (B, N) 模型输出的logit（未经过sigmoid）
        pos_item: (B, k1) 正样本索引，每个样本的正例数量可能不同（但输入格式固定为k1）
        neg_item: (B, k2) 负样本索引，每个样本的负例数量可能不同（但输入格式固定为k2）
    Returns:
        loss: 标量，平均后的BCE Loss
    """
    B = logit.shape[0]
    
    # --------------------------
    # 1. 提取正样本的logit和构造标签
    # --------------------------
    # 将pos_item从(B, k1)变形为(B*k1)，便于索引
    pos_flat = pos_item.view(-1)  # (B*k1,)
    # 生成批次索引：每个样本对应k1个位置，即[0,0,...,1,1,...,B-1,B-1]（共B*k1个）
    batch_idx = torch.arange(B, device=logit.device).unsqueeze(1).repeat(1, pos_item.shape[1]).view(-1)  # (B*k1,)
    # 提取正样本的logit：logit[batch_idx, pos_flat] -> (B*k1,)
    pos_logit = logit[batch_idx, pos_flat]
    # 正样本标签为1
    pos_label = torch.ones_like(pos_logit, device=logit.device)  # (B*k1,)
    
    # --------------------------
    # 2. 提取负样本的logit和构造标签
    # --------------------------
    neg_flat = neg_item.view(-1)  # (B*k2,)
    batch_idx_neg = torch.arange(B, device=logit.device).unsqueeze(1).repeat(1, neg_item.shape[1]).view(-1)  # (B*k2,)
    neg_logit = logit[batch_idx_neg, neg_flat]  # (B*k2,)
    neg_label = torch.zeros_like(neg_logit, device=logit.device)  # (B*k2,)
    
    # --------------------------
    # 3. 合并正负样本，计算BCE Loss
    # --------------------------
    # 合并logit和标签
    all_logit = torch.cat([pos_logit, neg_logit], dim=0)  # (B*k1 + B*k2,)
    all_label = torch.cat([pos_label, neg_label], dim=0)   # (B*k1 + B*k2,)
    
    # 计算BCE Loss（使用BCEWithLogitsLoss，内置sigmoid，数值更稳定）
    bce_loss = nn.BCEWithLogitsLoss()(all_logit, all_label)
    
    return bce_loss

def compute_metrics(eval_pred):
    # excepted all element in eval_pred in cuda 
    # 对于T5来说，eval_pred.predictions是一个元组，只有第一个才是真的，也就是logits[0]才是真的
    # 对于T5、llama来说，验证集早停就用loss来吧。
    logits, labels = eval_pred
    pos_items = labels[0]  # shape: (B, 1)
    logits = torch.as_tensor(logits)
    pos_items = torch.as_tensor(pos_items)
    
    ks = [5, 10, 20, 50]
    max_k = max(ks)
    
    _, predicted_indices = torch.topk(logits, k=max_k, dim=1)
    predicted_indices = predicted_indices.cpu().numpy()
    
    pos_items = pos_items.cpu().numpy().reshape(-1)
    
    results = {}
    results["samples"] = pos_items.shape[0]
    for k in ks:
        top_k_preds = predicted_indices[:, :k] # (B,k)
        
        pos_in_top_k = np.any(top_k_preds == pos_items[:, np.newaxis], axis=1)
        recall = pos_in_top_k.mean()
        results[f'recall@{k}'] = recall
        
        relevance = (top_k_preds == pos_items[:, np.newaxis]).astype(np.float32)
        
        ranks = np.arange(1, k + 1)  # 排名从1开始
        dcg_discount = 1.0 / np.log2(ranks + 1)  # 折扣因子
        dcg = np.sum(relevance * dcg_discount, axis=1)  # 按行求和
        
        idcg = np.ones_like(dcg)  
        
        ndcg = np.mean(dcg / idcg)
        results[f'ndcg@{k}'] = ndcg

    return results

class StatefulComputeMetrics:
    def __init__(self):
        self.ks = [5, 10, 20, 50]
        self.metrics_init()

    def metrics_init(self):
        self.metric = {}
        for k in self.ks:
            self.metric[f'recall@{k}'] = []
            self.metric[f'ndcg@{k}'] = []
        self.metric["samples"] = []

    def __call__(self, eval_pred, compute_result=False):
        # 对于T5来说，eval_pred.predictions是一个元组，只有第一个才是真的
        results = compute_metrics(eval_pred)
        for k,v in results.items():
            self.metric[k].append(v)

        if compute_result:
            weights = np.array(self.metric["samples"]) / sum(self.metric["samples"])
            results_all = {}
            for k,v in self.metric.items():
                if k != "samples":
                    results_all[k] = np.sum(np.array(v) * weights)
            self.metrics_init()
            return results_all
        else:
            return results

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        # print(f"1 -> {sentence} {trie_out}")
        return trie_out

    return prefix_allowed_tokens

def prefix_allowed_tokens_fn_lcrec(candidate_trie,start_token_list,tokenizer):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # print(f"0 -> {sentence} {tokenizer.decode(sentence)}")
        for i in range(len(sentence)-len(start_token_list),0,-1):
            if sentence[i:i+len(start_token_list)] == start_token_list:
                sentence = sentence[i:]
                break
        
        trie_out = candidate_trie.get(sentence)
        # print(f"1 ->  {trie_out}")
        return trie_out
    return prefix_allowed_tokens

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","").split("Response:")[-1] for _ in predictions]
    targets = [_.strip().replace(" ","") for _ in targets]
    # print(predictions)##################
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True) # 降序排列，由大到小。
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def ndcg_k(topk_results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg/len(topk_results)

def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit/len(topk_results)

def get_metrics_results(topk_results, metrics):
    res = {}
    res["samples"] = len(topk_results)
    for m in metrics:
        if m.lower().startswith("recall"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res

class generateComputeMetrics:
    def __init__(self,args,model,train_data,test_data,tokenizer):
        self.ks = [5, 10]
        self.metrics_init()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        tokenizer.add_tokens(test_data.get_new_tokens())
        all_items = test_data.get_all_items() | train_data.get_all_items()
        if args.model_code in ["tiger"]:
            self.candidate_trie = Trie(
                [
                    [0] + tokenizer.encode(candidate)
                    for candidate in all_items
                ]
            )
            self.prefix_allowed_tokens = prefix_allowed_tokens_fn(self.candidate_trie)
        elif args.model_code in ["lcrec"]:
            self.candidate_trie = Trie(
                [
                    tokenizer("Response:")["input_ids"][1:] + tokenizer.encode(candidate) + [tokenizer.eos_token_id]
                    for candidate in all_items
                ]
            )
            self.prefix_allowed_tokens = prefix_allowed_tokens_fn_lcrec(self.candidate_trie,tokenizer("Response:")["input_ids"][1:],tokenizer)
        self.all_items = all_items
        
        self.metrics = [f"recall@{k}" for k in self.ks] + [f"ndcg@{k}" for k in self.ks] 

    def model_generate_compute_metrics(self,input_ids,attention_mask,labels):
        with torch.no_grad():
            output = self.model.generate(
                        input_ids=input_ids, # torch.Size([4, 33])
                        attention_mask=attention_mask, # torch.Size([4, 33])
                        max_new_tokens=self.args.max_new_tokens, # 10
                        # max_length=10,
                        prefix_allowed_tokens_fn=self.prefix_allowed_tokens,
                        num_beams=self.args.num_beams, # 
                        num_return_sequences=self.args.num_beams, 
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                        top_p=1.0,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
            output_ids = output["sequences"]
            if self.args.num_beams==1:
                scores = torch.ones(output_ids.shape[0])
            else:
                scores = output["sequences_scores"]
            output_ = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
            labels  = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print(input_ids.shape)
            # 
            topk_res = get_topk_results(output_,scores,labels,self.args.num_beams)
            del output
            torch.cuda.empty_cache()
            gc.collect()

            
            return get_metrics_results(topk_res,self.metrics)

    def metrics_init(self):
        self.metric = {}
        for k in self.ks:
            self.metric[f'recall@{k}'] = []
            self.metric[f'ndcg@{k}'] = []
        self.metric["samples"] = []

    def __call__(self, eval_pred, compute_result=False):
        logits,labels,inputs = eval_pred
        results = self.model_generate_compute_metrics(inputs["input_ids"],
                    inputs["attention_mask"],inputs["labels"])
        for k,v in results.items():
            self.metric[k].append(v)

        if compute_result:
            weights = np.array(self.metric["samples"]) / sum(self.metric["samples"])
            results_all = {}
            for k,v in self.metric.items():
                if k != "samples":
                    results_all[k] = np.sum(np.array(v) * weights)
            self.metrics_init()
            return results_all
        else:
            return results
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    

)

def get_model_data(args):
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    train_data, valid_data = load_datasets(args)
    tokenizer = creat_tokenizer(args)
    if args.model_code in ["sasrec"]:
        args.vocab_size = train_data.item_num + 1
    elif args.model_code in ["lightgcn","mf","hgn","caser"]:
        args.n_users = train_data.user_num
        args.m_items = train_data.item_num
    elif args.model_code in ["tiger"]:
        tokenizer.add_tokens(train_data.get_new_tokens())
        args.vocab_size = len(tokenizer)
        tokenizer.save_pretrained(args.output_dir)
    elif args.model_code in ["lcrec"]:
        args.token_eos_id = tokenizer.eos_token_id
        tokenizer.add_tokens(train_data.dataset_.datasets[0].get_new_tokens())
        args.vocab_size = len(tokenizer)
        tokenizer.save_pretrained(args.output_dir)

    model = model_factory(args)
    # special process
    if args.model_code == "lightgcn":
        model.Graph = train_data.A.coalesce()
    elif args.model_code in ["tiger","lcrec"]:
        model.resize_token_embeddings(len(tokenizer))
        model.set_hyper(args,tokenizer)
    if args.model_code in ["lcrec"]:
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        # modules_to_save=args.lora_modules_to_save.split(","),
        trainable_token_indices=tokenizer.convert_tokens_to_ids(train_data.dataset_.datasets[0].get_new_tokens()),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
        model = get_peft_model(model, config)
        # model.get_output_embeddings().weight = model.get_input_embeddings().weight
        # todo resume training
        if local_rank == 0:
            model.print_trainable_parameters()

    if args.resume_from_checkpoint:
        if args.model_code in ["lcrec"]:
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.safetensors"
            )
            if args.task_code != "test":
                args.resume_from_checkpoint = False
            if os.path.exists(checkpoint_name):
                if local_rank == 0:
                    print(f"Restarting from {checkpoint_name}")
                from safetensors.torch import load_file
                adapters_weights = load_file(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                if local_rank == 0:
                    print(f"Checkpoint {checkpoint_name} not found")


    if local_rank == 0:
        print(vars(args))
        print(model)
    return model,train_data, valid_data,tokenizer

def unsqueeze_input(x):
    if len(x.shape)==1:
        x =x.unsqueeze(0)
    return x

def load_args():
    parser = argparse.ArgumentParser(description='LLM4REC')
    parser.add_argument("--config", type=str,default="/home/zhanghanlin/wangxu/acdemic_2025/config_file/zhl/lcrec_test_v1.yaml")
    parser.add_argument("--wandb_api_file", type=str,default="/home/zhanghanlin/wangxu/acdemic_2025/config_file/wandb_api.yaml")
    args = parser.parse_args()
    api_args = load_config(args.wandb_api_file)
    args = load_config(args.config)
    os.environ["WANDB_API_KEY"] = api_args.WANDB_API_KEY
    if not args.is_online:
        os.environ['WANDB_MODE'] = 'offline'
    else:
        wandb.login()
    os.environ["WANDB_PROJECT"]=args.project_name
    return args

def get_metric_fromargs(args):
    model,train_data, valid_data,tokenizer = get_model_data(args)
    trainer = creat_trainer(args,model,train_data,valid_data,tokenizer)

    if args.task_code in ["training","hp_search"]:
        model.config.use_cache = False
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.task_code == "test":
        # 重新load 模型 tokenizer...
        assert args.resume_from_checkpoint is not False
        if tokenizer is not None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    test_dataset = load_test_dataset(args)
    model.config.use_cache = True
    model.eval()
    with torch.no_grad(): 
        if args.model_code in ["tiger","lcrec"]:
            # trainer.args.prediction_loss_only = False
            compute_metrics_object = generateComputeMetrics(args,model,train_data,test_dataset,tokenizer)
            trainer.args.include_for_metrics = ["inputs"]
            trainer.args.prediction_loss_only = False
            trainer.compute_metrics = compute_metrics_object
            trainer.args.per_device_eval_batch_size =  args.per_device_test_batch_size
        if args.model_code == "lcrec":
            trainer.data_collator = LcrecTestCollator(args,tokenizer)
        _,_,metrics = trainer.predict(test_dataset)
        # trainer.log(metrics)
    return metrics

def optuna_args_change(trial,origin_args):
    args = copy.deepcopy(origin_args)
    des = ""
    for k,v in args.hp_search_cate.items():
        suggest_value = trial.suggest_categorical(k,v)
        setattr(args,k,suggest_value)
        des += f"{k}_({suggest_value})_"
    for k,v in args.hp_search_float.items():
        suggest_value = trial.suggest_float(k,v[0],v[1])
        setattr(args,k,suggest_value)
        des += f"{k}_({suggest_value})_"
    for k,v in args.hp_search_bind.items():
        tmp = eval(v)
        setattr(args,k,tmp)
        des += f"{k}_({suggest_value})"
    des = des[:-1]
    def replace_chars(text):
        for char in '() >/,\\[]':
            text = text.replace(char,'_')
        return text
    # des = replace_chars(des)
    args.project_name = args.task_name
    args.project_root = os.path.join(args.project_root,args.task_name)
    task_id = trial.number
    args.task_name = f"{args.project_name}_v{task_id}_{des}"
    args.output_dir = os.path.join(args.project_root,args.task_name)
    ensure_dir(args.output_dir)
    return args


def objective(trial,origin_args):
    args = optuna_args_change(trial,origin_args)
    wandb.init(project=args.project_name,name=args.task_name,config=args)
    metrics = get_metric_fromargs(args)
    wandb.finish()
    return metrics[args.hp_search_metric]


def hp_search(args):
    print(torch.cuda.is_available())
    set_seed(args.seed)
    args.output_dir = os.path.join(args.project_root,args.task_name)
    os.environ["WANDB_PROJECT"]=args.task_name
    ensure_dir(args.output_dir)
    os.chdir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    optuna_sql_path = os.path.join(args.output_dir,"sqlite")
    os.makedirs(optuna_sql_path, exist_ok=True)
    optuna_sql_db = os.path.join(optuna_sql_path,"optuna_studies.db")
    if args.grid_search:
        param_grid = {}
        for k,v in args.hp_search_cate.items():
            param_grid[k] = v
        study = optuna.create_study(study_name=args.task_name,pruner=optuna.pruners.NopPruner,
                direction=args.hp_search_direction,load_if_exists=True,
                storage=f'sqlite:///{optuna_sql_db}',sampler=optuna.samplers.GridSampler(param_grid))
    else:
        study = optuna.create_study(study_name=args.task_name,pruner=optuna.pruners.NopPruner,
                direction=args.hp_search_direction,load_if_exists=True,
                storage=f'sqlite:///{optuna_sql_db}')
    
    study.optimize(lambda trial:objective(trial,args),n_trials=args.hp_search_trial_num)
    
    print(f"please run : {BOLD}{PURPLE}wandb sync {os.path.abspath(os.path.join(args.output_dir,"wandb\\*"))}{RESET}")

