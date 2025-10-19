
import os
# wandb login

from transformers import EarlyStoppingCallback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import transformers
from baseline_config import *
from baseline_model import *
from utils import *

def all_in_one(args):
    if args.task_code == "hp_search":
        hp_search(args)
    else:
        train(args)


def train(args):
    print(torch.cuda.is_available())
    set_seed(args.seed)
    args.output_dir = os.path.join(args.project_root,args.task_name)
    ensure_dir(args.output_dir)
    os.chdir(args.output_dir)
    return get_metric_fromargs(args)


if __name__ == "__main__":
    args = load_args()
    all_in_one(args)
