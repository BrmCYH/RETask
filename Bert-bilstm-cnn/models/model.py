"""
A joint model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils import torch_utils, loader
from models import layers, submodel
from transformers import BertModel

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt):
        self.opt = opt
        # 初始化模型 主要模型 

        self.model = BiLSTMCNN(opt)
        # 主体评判函数 使用BCELoss
        # 客体评判函数 CrossEntropyLoss
        self.subj_criterion = nn.BCELoss(reduction='none')
        self.obj_criterion = nn.CrossEntropyLoss(reduction='none')
        #  
        # p in model.parameters 遍历模型的参数
        # 如果参数需要进行梯度更新，则加载到parameters列表中
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:# 是否使用cuda
            self.model.cuda()
            self.subj_criterion.cuda()
            self.obj_criterion.cuda()
        # 优化器 学习率优化器
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])
    

    def update(self, batch):# 调用update进行参数更新
        # 参数更新 
        """ Run a step of forward and backward model update. """
        # 数据加载
        if self.opt['cuda']:
            '''
                inputs 包括[]
            '''
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in batch[:3]]
            subj_start_binary = Variable(torch.LongTensor(batch[3]).cuda()).float()
            subj_end_binary = Variable(torch.LongTensor(batch[4]).cuda()).float()
            obj_start_relation = Variable(torch.LongTensor(batch[5]).cuda())
            obj_end_relation = Variable(torch.LongTensor(batch[6]).cuda())
            subj_start_type = Variable(torch.LongTensor(batch[7]).cuda())
            subj_end_type = Variable(torch.LongTensor(batch[8]).cuda())
            obj_start_type = Variable(torch.LongTensor(batch[9]).cuda())
            obj_end_type = Variable(torch.LongTensor(batch[10]).cuda())
            nearest_subj_start_position_for_each_token = Variable(torch.LongTensor(batch[11]).cuda())
            distance_to_nearest_subj_start = Variable(torch.LongTensor(batch[12]).cuda())# 实体起始位置标记
            distance_to_subj = Variable(torch.LongTensor(batch[13]).cuda())#  实体持续长度
            nearest_obj_start_position_for_each_token = Variable(torch.LongTensor(batch[14]).cuda())# 实体
            distance_to_nearest_obj_start = Variable(torch.LongTensor(batch[15]).cuda())
        else:
            inputs = [Variable(torch.LongTensor(b)) for b in batch[:3]]
            subj_start_label = Variable(torch.LongTensor(batch[3])).float()
            subj_end_label = Variable(torch.LongTensor(batch[4])).float()
            obj_start_label = Variable(torch.LongTensor(batch[5]))
            obj_end_label = Variable(torch.LongTensor(batch[6]))
            subj_type_start_label = Variable(torch.LongTensor(batch[7]))
            subj_type_end_label = Variable(torch.LongTensor(batch[8]))
            obj_type_start_label = Variable(torch.LongTensor(batch[9]))
            obj_type_end_label = Variable(torch.LongTensor(batch[10]))
            subj_nearest_start_for_each = Variable(torch.LongTensor(batch[12]))
            subj_distance_to_start = Variable(torch.LongTensor(batch[13]))
        
        
        mask = (inputs[0].data>0).float()# 输入掩码
        # step forward
        self.model.train()
        self.optimizer.zero_grad()


        # 模型中传入的参数太多了
        
        # 主体开始结束标记，客体开始结束标记  传入[T, C, Pos_tags, K1, K2]  mask   nearest_subj：实体1位置下标列表    distance_to:实体1到其他位置的距离   distance_to_subj:实体1 距离表现    nearest_obj:实体2位置下标列表            distance_to 实体2到其他位置的距离
        subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits = self.model(inputs, mask, nearest_subj_start_position_for_each_token, distance_to_nearest_subj_start, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)

        # 计算sub_start loss 均使用 crossentropy
        subj_start_loss = self.obj_criterion(subj_start_logits.view(-1, self.opt['num_subj_type']+1), subj_start_type.view(-1).squeeze()).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float()))/torch.sum(mask.float())
        
        subj_end_loss = self.obj_criterion(subj_end_logits.view(-1, self.opt['num_subj_type']+1), subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float()))/torch.sum(mask.float())
        
        obj_start_loss = self.obj_criterion(obj_start_logits.view(-1, self.opt['num_class']+1), obj_start_relation.view(-1).squeeze()).view_as(mask)
        obj_start_loss = torch.sum(obj_start_loss.mul(mask.float()))/torch.sum(mask.float())
        
        obj_end_loss = self.obj_criterion(obj_end_logits.view(-1, self.opt['num_class']+1), obj_end_relation.view(-1).squeeze()).view_as(mask)
        obj_end_loss = torch.sum(obj_end_loss.mul(mask.float()))/torch.sum(mask.float())
        # 计算总的loss
        loss = self.opt['subj_loss_weight']*(subj_start_loss + subj_end_loss) + (obj_start_loss + obj_end_loss)
        
        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val



    def predict_subj_per_instance(self, words):

        if self.opt['cuda']:
            words = Variable(torch.LongTensor(words).cuda())
            # chars = Variable(torch.LongTensor(chars).cuda())
            # pos_tags = Variable(torch.LongTensor(pos_tags).cuda())
        else:
            words = Variable(torch.LongTensor(words))
            features = Variable(torch.LongTensor(features))

        batch_size, seq_len = words.size()
        mask = (words.data>0).float()
        # forward
        self.model.eval()
        inputs, hidden, sentence_rep = self.model.based_encoder(words, mask)

        subj_start_logits, subj_start_outputs = self.model.subj_sublayer.predict_subj_start(hidden, sentence_rep, mask)

        _s1 = np.argmax(subj_start_logits, 1)
        
        nearest_subj_position_for_each_token, distance_to_nearest_subj =  loader.get_nearest_start_position([_s1])
        nearest_subj_position_for_each_token, distance_to_nearest_subj = Variable(torch.LongTensor(np.array(nearest_subj_position_for_each_token)).cuda()), Variable(torch.LongTensor(np.array(distance_to_nearest_subj)).cuda())

        subj_end_logits = self.model.subj_sublayer.predict_subj_end(subj_start_outputs, mask, nearest_subj_position_for_each_token, distance_to_nearest_subj, sentence_rep)
        
        return subj_start_logits, subj_end_logits, hidden, sentence_rep

    def predict_obj_per_instance(self, inputs, hidden, sentence_rep):

        if self.opt['cuda']:
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in inputs]
        else:
            inputs = [Variable(torch.LongTensor(b)).unsqueeze(0) for b in inputs[:4]]
        mask = (inputs[0].data>0).float()

        words, subj_start_position, subj_end_position, distance_to_subj = inputs # unpack

        self.model.eval()

        obj_start_logits, obj_start_outputs = self.model.obj_sublayer.predict_obj_start(hidden, sentence_rep, subj_start_position, subj_end_position, mask, distance_to_subj)

        _o1 = np.argmax(obj_start_logits, 1)
        nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start =  loader.get_nearest_start_position([_o1])
        nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start = Variable(torch.LongTensor(np.array(nearest_obj_start_position_for_each_token)).cuda()), Variable(torch.LongTensor(np.array(distance_to_nearest_obj_start)).cuda())


        obj_end_logits = self.model.obj_sublayer.predict_obj_end(obj_start_outputs, mask, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)

         
        return obj_start_logits, obj_end_logits






    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']
    def forward(self,batch):
        # 实现模型前向传播
        # self.model()
                # 数据加载
        if self.opt['cuda']:
            '''
                inputs 包括[]
            '''
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in batch[:3]]
            subj_start_binary = Variable(torch.LongTensor(batch[3]).cuda()).float()
            subj_end_binary = Variable(torch.LongTensor(batch[4]).cuda()).float()
            obj_start_relation = Variable(torch.LongTensor(batch[5]).cuda())
            obj_end_relation = Variable(torch.LongTensor(batch[6]).cuda())
            subj_start_type = Variable(torch.LongTensor(batch[7]).cuda())
            subj_end_type = Variable(torch.LongTensor(batch[8]).cuda())
            obj_start_type = Variable(torch.LongTensor(batch[9]).cuda())
            obj_end_type = Variable(torch.LongTensor(batch[10]).cuda())
            nearest_subj_start_position_for_each_token = Variable(torch.LongTensor(batch[11]).cuda())
            distance_to_nearest_subj_start = Variable(torch.LongTensor(batch[12]).cuda())# 实体起始位置标记
            distance_to_subj = Variable(torch.LongTensor(batch[13]).cuda())#  实体持续长度
            nearest_obj_start_position_for_each_token = Variable(torch.LongTensor(batch[14]).cuda())# 实体
            distance_to_nearest_obj_start = Variable(torch.LongTensor(batch[15]).cuda())
        else:
            inputs = [Variable(torch.LongTensor(b)) for b in batch[:3]]
            subj_start_label = Variable(torch.LongTensor(batch[3])).float()
            subj_end_label = Variable(torch.LongTensor(batch[4])).float()
            obj_start_label = Variable(torch.LongTensor(batch[5]))
            obj_end_label = Variable(torch.LongTensor(batch[6]))
            subj_type_start_label = Variable(torch.LongTensor(batch[7]))
            subj_type_end_label = Variable(torch.LongTensor(batch[8]))
            obj_type_start_label = Variable(torch.LongTensor(batch[9]))
            obj_type_end_label = Variable(torch.LongTensor(batch[10]))
            subj_nearest_start_for_each = Variable(torch.LongTensor(batch[12]))
            subj_distance_to_start = Variable(torch.LongTensor(batch[13]))
        mask = (inputs[0].data>0).float()# 输入掩码
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits = self.model(inputs, mask, nearest_subj_start_position_for_each_token, distance_to_nearest_subj_start, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)
        # 模型前向传播方法
        return subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits
        # pass
from transformers import BertConfig
class BiLSTMCNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(BiLSTMCNN, self).__init__()
        self.pretrainedModel=BertModel.from_pretrained(opt["bertmodel"])
        for param in self.pretrainedModel.embeddings.parameters():
            param.requires_grad = False
        self.config=BertConfig.from_pretrained(opt["bertmodel"])
        self.drop = nn.Dropout(opt['dropout'])
        # self.word_emb = nn.Embedding(opt['word_vocab_size'], opt['word_emb_dim'], padding_idx=0)
        # #
        # self.char_emb = nn.Embedding(opt['char_vocab_size'], opt['char_emb_dim'], padding_idx=0)

        # self.pos_emb = nn.Embedding(opt['pos_size'], opt['pos_emb_dim'], padding_idx=0)
        
        self.input_size = self.config.output_hidden_states
        # 双向LSTM
        self.rnn = nn.LSTM(self.input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'], bidirectional=True)
        # 实体1，2子层
        
        self.subj_sublayer = submodel.SubjTypeModel(opt, 4*opt['hidden_dim'], 2*opt['hidden_dim'],self.pretrainedModel.config.max_position_embeddings)
        self.obj_sublayer = submodel.ObjBaseModel(opt, 8*opt['hidden_dim'], 2*opt['hidden_dim'],self.pretrainedModel.config.max_position_embeddings)
        # Char编码层
        # self.char_encoder = layers.CharEncoder(opt)
        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        # self.init_weights()
    
    def init_weights(self):
        #  对模型不同的嵌入层进行初始化 word_emb,char_emb,Pos_emb
        if self.emb_matrix is None:
            self.word_emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
        #     self.word_emb.weight.data.copy_(self.emb_matrix)
        # self.char_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        # self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)


  
        # decide finetuning
        # 是否要微调词嵌入层 或 微调词嵌入层中多少词
        if self.topn <= 0:# 
            print("Do not finetune word embedding layer.")
            self.word_emb.weight.requires_grad = False
            # 微调前n层词嵌入
        elif self.topn < self.opt['word_vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.word_emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        '''
        用于初始化LSTM模型的隐藏状态和记忆状态
        h0,c0分别代表隐藏层和记忆状态
        state_shape=(2*Lstm层数，批次大小，隐藏层维度)
        '''
        state_shape = (2*self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)#创建全0的张量 同时赋予两个变量
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def based_encoder(self, words,mask):
        '''
            seq_lens为batch中每个sample的长度：列表
        '''
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
        
        batch_size,_ = words.size()# 传入一个batch 的数据
        
        # embedding lookup
        # word_inputs = self.drop(self.word_emb(words))# 使用dropout 
        # pos_inputs = self.pos_emb(pos_tags)# 位置标签编码
        # chars = chars.view(-1,15)
        # chars_mask = (chars.data>0).float()#
        # 调用 对char进行encoder
        # char_inputs = self.char_encoder(self.drop(self.char_emb(chars)), chars_mask).contiguous().view(batch_size,seq_len,self.opt['char_hidden_dim'])
        word_inputs=[item for item in words]
        # 模型输入
        inputs = [word_inputs]# 
        
        inputs = self.drop(torch.cat(inputs, dim=2))

        #LSTM模型在输入数据和输出数据之前进行了压缩 

        # 获得初始的h0和c0
        h0, c0 = self.zero_state(batch_size)# 调用zero_state
        # 将数据打包压缩成一个序列  input为输入数据、seq_lens batch_size大小的长度列表
        # 压缩1

        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        # 传入压缩序列和隐藏层，记忆层状态    LSTM的输出信息hidden包括所有时间步骤的输出结果(batch_size,hidden_dim)*all_steps
        hidden, (ht, ct) = self.rnn(packed_inputs, (h0, c0))

        # 解压缩2 将LSTM输出的数据解压缩为原始的形状  (batch_size,hidden_dim)--->(batch_size,max_seq_len,hidden_dim)
        hidden, output_lens = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        # 根据句子的掩码mask 在hidden中的不属于句子部分(1-mask)的元素设置为一个极小的值 1e10
        hidden_masked = hidden - ((1 - mask) * 1e10).unsqueeze(2).repeat(1,1,hidden.shape[2]).float()
        # 整个句子的表示 sentence_rep 句子表示  
        # 对hidden_masked进行池化操作  在hidden_masked的第二个维度上进行最大池化操作
        # torch.transpose 完成对维度的交换
        # 池化得到(batch_size,hidden_dim,1)
        sentence_rep = F.max_pool1d(torch.transpose(hidden_masked, 1, 2), hidden_masked.size(1)).squeeze(2)#squeeze(2)将最后一个维度去掉
        return inputs, hidden, sentence_rep
    
    def forward(self, inputs, mask, nearest_subj_position_for_each_token, distance_to_nearest_subj, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start):

        words,subj_start_position, subj_end_position = inputs # unpack
        batch_size, seq_len = words.size()
        # encoder 调用based_encoder进行嵌入
        # 第一层 传入词和掩码
        inputs, hidden, sentence_rep = self.based_encoder(words,  mask)
        # 下游子层判断 实体1，2的logits
        # 判断实体 以及关系 传入hidden ，句子总结信息，
        subj_start_logits, subj_end_logits = self.subj_sublayer(hidden, sentence_rep, mask, nearest_subj_position_for_each_token, distance_to_nearest_subj)
        obj_start_logits, obj_end_logits = self.obj_sublayer(hidden, sentence_rep, subj_start_position, subj_end_position, mask, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)

        
        return subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits
    

