pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 # 2.7.0 cuda 12.6
pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.52.3  -i https://pypi.tuna.tsinghua.edu.cn/simple # 4.52.3
pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple # 0.48.1
pip install cachetools -i https://pypi.tuna.tsinghua.edu.cn/simple # 5.3.2
pip install optuna -i https://pypi.tuna.tsinghua.edu.cn/simple # 3.4.0
pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple # 0.5.0
# 注意将transformer库中的 /home/zhanghanlin/anaconda3/envs/wx_env/lib/python3.12/site-packages/transformers/trainer.py 中的 evaluation_loop 函数中的
# if not self.args.batch_eval_metrics or description == "Prediction":
# 替换为
# if not self.args.batch_eval_metrics : ，也就是把所有的description == Prediction都去掉。 ！！！！！！！这个非常重要，会导致oom