import json
import numpy as np
import random
from random import choice
from tqdm import tqdm
import collections

global num
# 计算
import os,sys,copy
from transformers import BertTokenizer
class Data:
    '''
    args:
        bert_directory :用于初始化分词器
        train_file :训练集路径
        valid_file
        test_file
    '''
    def __init__(self) -> None:
        # self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False) # 关系词表的设置

        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.weight = {}
    def show_data_summary(self): #
        print("DATA SUMMARY START:")        # 
        # print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()
    def generate_instance(self, args, data_process):
        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=False)
        # 读入
        if "train_file" in args:
            self.train_loader = data_process(args.train_file, self.relational_alphabet, tokenizer)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
        if "valid_file" in args:
            self.valid_loader = data_process(args.valid_file, self.relational_alphabet, tokenizer)
        if "test_file" in args:
            self.test_loader = data_process(args.test_file, self.relational_alphabet, tokenizer)
        #
        self.relational_alphabet.close()
def data_process(input_doc, relational_alphabet, tokenizer):
    samples = []
    with open(input_doc) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
    for i in range(len(lines)):
        token_sent = [tokenizer.cls_token] + tokenizer.tokenize(remove_accents(lines[i]["sentText"])) + [tokenizer.sep_token]
        triples = lines[i]["relationMentions"]
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": []}
        for triple in triples:
            head_entity = remove_accents(triple["em1Text"])
            tail_entity = remove_accents(triple["em2Text"])
            head_token = tokenizer.tokenize(head_entity)
            tail_token = tokenizer.tokenize(tail_entity)
            relation_id = relational_alphabet.get_index(triple["label"])
            head_start_index, head_end_index = list_index(head_token, token_sent)
            assert head_end_index >= head_start_index
            tail_start_index, tail_end_index = list_index(tail_token, token_sent)
            assert tail_end_index >= tail_start_index
            target["relation"].append(relation_id)
            target["head_start_index"].append(head_start_index)
            target["head_end_index"].append(head_end_index)
            target["tail_start_index"].append(tail_start_index)
            target["tail_end_index"].append(tail_end_index)
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        samples.append([i, sent_id, target])
    print(lines[0])
    print(samples[0])
    return samples

def build_data(args):
    '''
    创建pickle二进制数据集
    
    '''
    file=args.generated_data_directory +args.dataset_name+"_"+args.model_name+"_data.pickle"# 创建处理的二进制文件
    if os.path.exists(file) and not args.refresh: #如果二进制文件存在 并且无需更新
        data =load_data_setting(args)# 直接文件中加载
    else:
        data=Data()
        data.generate_instance(args,data_process)
        save_data_setting(data,args)        # 保存文件
def get_nearest_start_position(S1):#传入内容为 batch[4] s1（记录实体1、2起始位置的列表）

    nearest_start_pos = []
    current_start_pos = 0
    current_pos = []
    flag = False
    # 遍历一条记录
    for i, start_label in enumerate(S1):
        if start_label > 0:# 如果这个位置存在实体
            current_start_pos = i # current_start_pos 存入这个位置的下标
            flag = True
        nearest_start_pos.append(current_start_pos)#  记录每个实体的位置
        if flag > 0:# 如果已经存在实体 并且 记录距离 超过10 则记为499
            #
            if i-current_start_pos > 10:#判断他们之间的距离是否超过10 
                current_pos.append(499)
            else:
                current_pos.append(i-current_start_pos)#否则 # currentpos追加
        else:# 没遇到实体则记录为499
            current_pos.append(499)
    # print(start_pos_list)
    # print(nearest_start_pos)
    # print(current_pos)
    # print('-----')
    # nearest_start_list.append(nearest_start_pos)# 记录每条记录中  实体的下标
    # current_distance_list.append(current_pos)# 记录每条记录的 pos距离
    return nearest_start_pos, current_pos# 

def locate_entity(token_list, entity):
    try:
        for i, token in enumerate(token_list):
            if entity.startswith(token):
                len_ = len(token)
                j = i+1 ; joined_tokens = [token]
                while len_ < len(entity):
                    len_ += len(token_list[j])
                    joined_tokens.append(token_list[j])
                    j = j+1
                if ''.join(joined_tokens) == entity:
                    return i, len(joined_tokens)
    except Exception:
        # print(entity,token_list)
        pass
    return -1, -1

def seq_padding(X,ML):#对一个批次的信息进行填充
    # L = [len(x) for x in X] # 创建序列
    # # ML =  config.MAX_LEN #max(L)
    # ML = max(L)#
    return [X + [0] * (ML - len(X))]

def char_padding(X):
    L_S = [len(x) for x in X]
    ML_S = max(L_S)
    L = [[len(t) for t in s] for s in X]
    # ML =  config.MAX_LEN #max(L)
    ML = max([max(l) for l in L])
    if ML <= 15:
        return [[t + [0] * (15 - len(t)) for t in s]+[[1] * 15 for i in range(ML_S-len(s))] for s in X]
    else:
        return [[t + [0] * (15 - len(t)) if len(t) <=15 else t[:15] for t in s]+[[1] * 15 for i in range(ML_S-len(s))] for s in X]
def get_pos_tags(tokens, pos_tags, pos2id):
    pos_labels = [pos2id.get(flag,1) for flag in pos_tags]
    #将pos 转为id
    if len(pos_labels) != len(tokens):
        #如果 pos_labels 与tokens不等长
        print(pos_labels)
        return False
    return pos_labels

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    '''
    get_positions(2, 5, 10)
    [-2, -1, 0, 0, 0, 0, 1, 2, 3, 4]
    '''
    # 起始
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))
    # list[-start_idx,0,0,..,length-end_idx] 整个列表长度为tokens的长度

def sort_all(batch, lens):# batch 以及对应的 元素长度
    """ Sort all fields by descending order of lens, and return the original indices. """
    # [lens],[range],[batch]
    # (lens[0],[range][0],batch[0])
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    # sorted() 根据lens进行排序  得到[(lens[x],[range][x],batch[x]),....]的列表
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    # sorted_all=[(lens[x],index[x],batch[x]),...]
    return sorted_all[2:], sorted_all[1]

def sort_key(item):
    return (item[0], item[1])

#  数据集对象
from torch.utils.data import Dataset
class RE(Dataset):
    def __init__(self,data, predicate2id, tokenizer, subj_type2id, obj_type2id,):
        # self.batch_size = batch_size
        
        self.predicate2id = predicate2id
        # self.char2id = char2id
        # self.pos2id = pos2id
        self.tokenizer=tokenizer
        self.subj_type2id = subj_type2id
        self.obj_type2id = obj_type2id
        self.data,self.Max_length=self.preprocess(data)
        
    def preprocess(self, data):#--> processes[Tuple(List),Tuple(List)]

        '''
            处理数据 将每条记录的信息转为 一条元组

            tokens=context信息
            subj_type{(起,止):idlist[Type1,]}
            item[(起,止):[(起,止,relaType)]]  item中key是从1开始
            obj_type{(起,止):idlist[Type]}
            s1:以1标记实体的起始位置
            s2:以1标记实体的结束位置
            Ts1: 以Type_id标记实体的起始位置
            Ts2：以Type_id标记实体的结束位置
            o1: item中随机选择的实体1 以1标记实体起始位置
            o2: item中随机选择的实体1 以1标记实体结束位置
            To1： 随机选择的实体1 以obj_type中实体类型进行标记
            To2:  标记结束位置
            distance_to_subj=get_positions(2, 5, 10) -->return [-2, -1, 0, 0, 0, 0, 1, 2, 3, 4]

        '''
        max_len=[]# 记录数据集最大样本长度
        processed = []
        for d in data:
            tokens1 = d['tokens']#info
            items = {}
            tokens_ids = [self.tokenizer.tokenize(word) for word in tokens1]
            '''
            下面完成实体标记 更新
            '''
            # 记录实体
            entity_list=[]
            for sp in d["spo_details"]:
                if (sp[0],sp[1]) not in entity_list:
                    entity_list.append(tuple((sp[0],sp[1])))
                if (sp[4],sp[5]) not in entity_list:
                    entity_list.append(tuple((sp[4],sp[5])))
            # 记录对应下标
            NER={}
            for item in entity_list:
                ownlist=[]
                # youbiao=0
                
                for i,per in enumerate(tokens_ids,start=0):
                    # 如果i和当前实体的开始内容相同
                    # print(youbiao)

                    # if  youbiao==-1:# 直接添加后面的信息
                    #     ownlist.extend(per)
                    if i<=item[0]:#比较起始位置
                        # ownlist.extend(per)

                        if i==item[0]:
                            qishi=len(ownlist)
                            # 实体游标之前的内容
                        else:# 游标只有在遍历完列表后才会为-1  所以一般不会更新
                            qishi=len(ownlist)-1# 获得实体的起始位置
                        ownlist.extend(per)#在添加内容
                    elif i>=item[0]and i<=item[1]:# 遍历元素在起始位置和结束之间

                        ownlist.extend(per)
                        jieshu=len(ownlist)-1
                        if i==item[1]:#
                            #判断是否遍历到了实体的最后一个内容
                            # 如果entity_list还有实体 则继续向后；否则赋值为-1 表示没有实体 后续直接追加
                            NER[(item[0],item[1])]=(qishi,jieshu)
                            # youbiao+=1 if len(item)-1>youbiao else -1
                    if i>item[1]:
                        break
            # 获得更新后的列表
            for per in tokens_ids:
                
                ownlist.extend(per)

            subj_type = collections.defaultdict(list)
            obj_type = collections.defaultdict(list)
            # 获取所有的实体 遍历spo_details获取所有实体
            for sp in d['spo_details']:#spo_detail [[起始，类型，关系，尾实体起始，类型]]
                '''
                "spo_details": [
                [14, 16, "PERSON", "/people/person/nationality", 35, 36, "LOCATION"], 
                [35, 36, "LOCATION", "/location/country/capital", 33, 34, "LOCATION"], 
                [35, 36, "LOCATION", "/location/location/contains", 33, 34, "LOCATION"], 
                [14, 16, "PERSON", "/people/deceased_person/place_of_death", 33, 34, "LOCATION"]], 
                '''          
                key = NER[(sp[0], sp[1])]#关系sp的首实体 起止下表(起，尾)
                subj_type[key].append(self.subj_type2id[sp[2]])# 首实体追加其typeid
                # items字典中创建key
                if key not in items:
                    items[key] = []
                # 如果这个实体没有在items的实体 列表中 则创建新列表
                # items[key]追加其 关系客体 及关系id（起，尾，关系id）
                # 如果 key对应多个关系客体，则将其他关系客体追加
                
                items[key].append((NER[(sp[4],sp[5])][0],
                                    NER[(sp[4],sp[5])][1],
                                    self.predicate2id[sp[3]]))
                # 关系客体的列表 追加 客体属性
                obj_type[(NER[(sp[4],sp[5])][0],NER[(sp[4],sp[5])][1])].append(self.obj_type2id[sp[6]])


            # 如果该条记录存在 实体关系      
            if items: #（s_idx,s_idx）:(o_idx,o_idx,s_o_rela)


                s1, s2 = [0] * len(ownlist), [0] * len(ownlist)
                ts1, ts2 = [0] * len(ownlist), [0] * len(ownlist)
                # 遍历items字典的主实体
                # 遍历字典的key
                # 将所有的实体对象 放入到stp中
                for j in items:
                    #
                    # for x in range(j[0],j[1]+1):
                    #     s1[x] = 1
                    s1[j[0]] = 1
                    s2[j[1]] = 1
                    stp = choice(subj_type[j]) # sub_type记录了实体名称对应的实体类型列表  choice 随机选择一个列表项
                    ts1[j[0]] = stp  
                    ts2[j[1]] = stp
                # k1,k2为下标 记录实体的起止
                k1, k2 = choice(list(items.keys())) #  从字典items键中随机选择一个键赋值给 k1，k2
                # o1,o2 列表在关系实体对应下标位置填入关系类型
                o1, o2 = [0] * len(ownlist), [0] * len(ownlist) 
                to1, to2 = [0] * len(ownlist), [0] * len(ownlist) 
                # 得到一个列表  0的部分为实体，之前内容为-1递增，后面为1递增
                distance_to_subj = get_positions(k1, k2, len(ownlist))
                # 遍历items字典的 客体
                for j in items[(k1, k2)]:# 遍历k1，k2元素的所有 关系实体B
                    # print(len(o1))
                    
                    o1[j[0]] = j[2] # #relation 标注
                    o2[j[1]] = j[2] #
                    otp = choice(obj_type[(j[0], j[1])])# otp选择 该下表对应的实体类型 在obj_type列表中
                    to1[j[0]] = otp # 实体2类型
                    to2[j[1]] = otp
                        #        0       1     2      3  4   5   6    7    8   9     10    11
                # processed=[]
                processed += [(ownlist, [k1], [k2-1], s1, s2, o1, o2, ts1, ts2, to1, to2, distance_to_subj)]
                    #            0           1           2    3      4    5   6  7    8    9    10   11   12 
                #processed += [(tokens_ids, chars_ids, [k1], [k2-1], s1, s2, o1, o2, ts1, ts2, to1, to2, pos_tags, distance_to_subj)]
            max_len +=[len(ownlist)]
            print(self.tokenizer.decode(ownlist))
            break
        if max(max_len)<128:
            Max_length=128
        elif max(max_len)<256:
            Max_length=256
        else:
            Max_length=512
        return processed,Max_length
    

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        # 获取一条记录
        one = self.data[index]# （Batch=[Tuple,...]）

        # batch_size = len(batch)#
        # batch = list(zip(*batch))

        assert len(one) == 12  #batch中每个列表的长度为14
        #list zip 组合成一个列表中的12个元组对象
        
        
        # lens = [len(x) for x in batch[0]]# 获得batch中每个序列的长度

        # batch, orig_idx = sort_all(batch, lens)# According to length For sorting samples
        # 序列填充
        T = np.array(seq_padding(one[0],self.Max_length))# token seq
        # C = np.array(char_padding(batch[1])) # char seq
        K1, K2 = np.array(one[1],self.Max_length), np.array(one[2],self.Max_length) #k1,k2 起止列表
        
        
        S1 = np.array(seq_padding(one[3],self.Max_length))#
        S2 = np.array(seq_padding(one[4],self.Max_length))
        O1 = np.array(seq_padding(one[5],self.Max_length))
        O2 = np.array(seq_padding(one[6],self.Max_length))
        TS1 = np.array(seq_padding(one[7],self.Max_length))
        TS2 = np.array(seq_padding(one[8],self.Max_length))
        TO1 = np.array(seq_padding(one[9],self.Max_length))
        TO2 = np.array(seq_padding(one[10],self.Max_length))

        # Pos_tags = np.array(seq_padding(batch[12])) #  词性序列 填充
        # Distance_to_subj = np.array(seq_padding(one[1],self.Max_length))   # distance_to_subj  [token_length -x ,+len-x]
        # # 获得实体s1的下标 以及实体s1到其他位置的距离表示
        # Nearest_S1, Distance_S1 = get_nearest_start_position(one[3]) # 实体1的起始位置
        # # Nearest_S1 实体位置列表  实体距离列表
        # Nearest_S1, Distance_S1 = np.array(seq_padding(Nearest_S1,self.Max_length)), np.array(seq_padding(Distance_S1,self.Max_length))
        # Nearest_O1, Distance_O1 = get_nearest_start_position(one[5])# 实体2的起始位置
        # Nearest_O1, Distance_O1 = np.array(seq_padding(Nearest_O1,self.Max_length)), np.array(seq_padding(Distance_O1,self.Max_length))
        #  0          1           2      3     4    5   6   7  8     8     9   10  11  12         13 
        #[(tokens_ids, chars_ids, [k1], [k2-1], s1, s2, o1, o2, ts1, ts2, to1, to2, pos_tags, distance_to_subj)]
            #   0  1   2    3   4   5   6    7   8     9   10   11           1 2          13                14            15       16
        return (T, S1, S2,TO1, TO2)
        # item (sentences,)


        # return super().__getitem__(index)