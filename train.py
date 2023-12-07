"""
Train a model on for JointER.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from transformers import Trainer,TrainingArguments
from typing import Any, Dict
from transformers import DataCollator,DataCollatorWithPadding
from dataset.build import *
from torch.utils.data import random_split,DataLoader
import torch_utils
class RETrainer(Trainer):
    def __init__(self):
        super.__init__()
        self.subj_criterion = nn.BCELoss(reduction='none')
        self.obj_criterion = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self,model,inputs):#计算模型损失
        subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits=model(inputs)
        loss
        # parameters = [p for p in model.parameters() if p.requires_grad]
        # if opt['cuda']:# 是否使用cuda
        #     model.cuda()
        
            # subj_criterion.cuda()
            # obj_criterion.cuda()

        pass

# class Data2Collator(DataCollator):
#     def __call__(self,features):
#         pass
# 指标计算方法
def compute_metrics()->Dict:
    pass
class Metric:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        pass
parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir",type=str,default="cache")
parser.add_argument('--data_dir', type=str, default='dataset/NYT-multi/data')
# parser.add_argument('--vocab_dir', type=str, default='dataset/NYT-multi/vocab')
# parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding dimension.')
# parser.add_argument('--char_emb_dim', type=int, default=100, help='Char embedding dimension.')
# parser.add_argument('--pos_emb_dim', type=int, default=20, help='Part-of-speech embedding dimension.')
# parser.add_argument('--position_emb_dim', type=int, default=20, help='Position embedding dimension.')
parser.add_argument('--bertmodel',type=str,default="bert-base-uncased")
parser.add_argument('--hidden_dim', type=int, default=768, help='RNN hidden state size.')
# parser.add_argument('--char_hidden_dim', type=int, default=100, help='Char-CNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=1, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.4, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
# parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--subj_loss_weight', type=float, default=1, help='Subjct loss weight.')
parser.add_argument('--type_loss_weight', type=float, default=1, help='Object loss weight.')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adamw', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--load_saved', type=str, default='')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=400, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
# parser.add_argument('--logger_dir',type=str,default="logging")
parser.add_argument('--position_emb_dim',type=int,default=768)
parser.add_argument('--seed', type=int, default=35)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())# ruguo 有GPU则使用
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument("--dir",type=str,default="D:\\workstation\\NERWork\\RETask\\RE\\RETask-main\\RETask-main\\Bert-bilstm-cnn\\dataset\\NYT-multi\\data")
parser.add_argument('--LocalModel',type=str,default="D:\\workstation\\NERWork\\RETask\\RE\\bert")
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument("--combine",type=bool,default=True)# 合并
parser.add_argument('--load_method',type=str,default="local")
parser.add_argument('--val_size',type=float,default=0.2)
args = parser.parse_args()




torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

#
from utils.loader import DataLoader,RE # 数据集加载
from models.model import RelationModel,BiLSTMCNN
from utils import helper, score
from utils.vocab import Vocab #词表
from transformers import BertConfig,BertModel,BertTokenizer

#以字典形式返回args
opt = vars(args)
if opt['gpu']:
    device="cuda"
else:
    device='cpu'
# import pickle
from utils.loader import RE
# load data
    # 分词器加载

print("Loading data from {} with batch size {}...".format(opt['data_dir'],opt["batch_size"]))

if opt['load_method']=="local":
    tokenizer=BertTokenizer.from_pretrained(opt["LocalModel"])
else:
    tokenizer=BertTokenizer.from_pretrained(opt['bertmodel'])

# 如果本地文件存在 则加载本地文件 否则
reartype=".json"
opt["datafiles"]=os.path.join(opt["dir"],"{}_{}{}".format(opt["dir"].split("\\")[-2],opt["LocalModel"].split('\\')[-1],reartype))

if os.path.exists(opt["datafiles"]):
    # 调用RE类加载模型
    dataset=RE(data_path=opt['datafiles'],tokenizer=tokenizer,device=device)
    # data=json.load()

else:
    filename=build_data(opt,tokenizer)# 进行文件生成
    #调用 加载模型
    dataset=RE(data_path=os.path.join(opt['dir'],filename),tokenizer=tokenizer,device=device)

# 数据集划分
val_size=int(len(dataset)*opt['val_size'])
train_size=len(dataset)-int(len(dataset)*opt['val_size'])

train_dataset, val_dataset = random_split(dataset, [train_size,val_size])

# 创建数据集加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# model save
model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']# 模型id
model_save_dir = opt['save_dir'] + '/' + model_id  #
# 保存模型
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)


# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
# logger 路径
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\dev_p\tdev_r\tdev_f1")
# print model info
helper.print_config(opt)
print(opt['num_class']) # 类别数量
# 
# init model  
model = RelationModel(opt)# embdlayer--> hiddensize 模型初始化
# create trainer
# Train Arguments
# if opt['gpu']:
model.to(device)


training_args=TrainingArguments(
    output_dir=opt["save_dir"],
    num_train_epochs=opt['num_epoch'],
    per_device_train_batch_size=16,
    learning_rate=opt['lr'],
    weight_decay=opt['weight_decay'],
    logging_dir=model_save_dir + '/' + opt['log'],
    save_steps=opt["save_epoch"],
    save_strategy="epoch",
    evaluation_strategy='epoch',
    device=device
)
# optimizer
# 优化器 学习率优化器
optimizer = torch_utils.get_optimizer(opt['optim'], model.parameters, opt['lr'], opt['weight_decay'])
#  
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    compute_metrics=Metric,
    optimizers=optimizer
)
trainer.train()# 模型训练
# 调用模型forward 进行模型评估
# 此处与trainer中的验证集训练冲突  是用于在训练之外对模型进行单独的评估操作
trainer.evaluate()
# 

if opt['load_saved'] != '':# 加载 bestmodel
    model.load(opt['save_dir']+'/'+opt['load_saved']+'/best_model.pt')
dev_f1_history = []
current_lr = opt['lr']


global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

max_steps = len(train_batch) * opt['num_epoch']


# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        # 调用model.update 进行损失计算
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))




    # 如何进行验证集实现
    # eval on dev
    # print("Evaluating on dev set...")
    # dev_f1, dev_p, dev_r, results = score.evaluate(dev_batch[epoch], model)

    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    # best_f1 = dev_f1 if epoch == 1 or dev_f1 > max(dev_f1_history) else max(dev_f1_history)
    print("epoch {}: train_loss = {:.6f}".format(epoch,\
            train_loss))
    file_logger.log("{}\t{:.6f}".format(epoch, train_loss))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    # if epoch == 1 or dev_f1 > max(dev_f1_history):
    #     copyfile(model_file, model_save_dir + '/best_model.pt')
    #     print("new best model saved.")
    #     with open(model_save_dir + '/best_dev_results.json', 'w') as fw:
    #         json.dump(results, fw, indent=4, ensure_ascii=False)
    #     print("new best results saved.")
    # if epoch % opt['save_epoch'] != 0:
    #     os.remove(model_file)
    
    # # lr schedule
    # if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
    #         opt['optim'] in ['sgd', 'adagrad']:
    #     current_lr *= opt['lr_decay']
    #     model.update_lr(current_lr)

    # dev_f1_history += [dev_f1]
    # print("")

print("Training ended with {} epochs.".format(epoch))

