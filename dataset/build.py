import json,os
from typing import Dict
count = 0
def posi(list1,list2):
    '''
    判断list2的[1:-2] 在list1的下标位置
    '''
#     print(list1[1:-1])
    itm=list2[1:-1]#肯定包含开始和结束标记 # 不包含-1
#     print(itm)
#     print(itm[0])
    eco=list1.index(itm[0])
    ovo=list1.index(itm[-1])
#     print(list1[eco:ovo+1])
    if len(itm)==1:
        # print("长度为1",eco,ovo)
        return eco,eco+1
    if list1[eco:ovo+1]==itm:
        # print("直接在一块",eco,ovo)
        return eco,ovo+1
    else:
        
        fal={"head":0,"rear":0}# 记录开始和结束下标
        for im,j in enumerate(range(eco,len(list1)),start=eco):# 遍历剩余的
            
            if list1[im]==itm[0] and fal['head']==0:# 判断如果当前值与itm[0]相同 往下进行判断 并且 
                #在itm的第二项开始判断 im下标处于list1[im]值相同 所以b从im+1开始
                fal['head']=im
                # 循环逻辑 判断当前是否是一个连续序列存在于接下来

                for b,x in enumerate(itm[1:],start=im+1):
                    # 如果当前值相同
                    if x ==list1[b]:#
                        fal['rear']=b
                        continue
                    else:
                        # x !=list1[b] and fal['head']==im:# 如果有值不匹配 且 本次以im开头 则设置fal head 为0
                        fal['head']=0
                        fal['rear']=0 
                        break
                        # continue
                if x==itm[-1] and fal['head']!=0:
                    fal['rear']=b
                    
                # if fal['rear']==b and fal["head"]==im:
                #     return 
    if list1[fal["head"]:fal['rear']+1]==itm:
        # print(f'多个匹配{fal["head"]},{fal["rear"]+1}')
        return fal["head"],fal['rear']+1
    else:
        print("ERROR")
        return 0,0      
    # pass

# count=0
def chuli(oner:Dict,tokenizer,opt)->Dict:
    # 实现数据下标的转换

    # 对内容进行分词

    txt=tokenizer.encode(" ".join(oner["tokens"]))
    # print(txt)
    listone=[]

    for item in oner["spo_details"]:
        # print(item)
        ett1=" ".join(oner["tokens"][item[0]:item[1]])
        ett2=" ".join(oner["tokens"][item[4]:item[5]])
        em1=tokenizer.encode(ett1)
        em2=tokenizer.encode(ett2)
        eca_1,out_1=posi(txt,em1)
        # print(type(eca_1))
        

        e1_type_id=opt["subj_type2id"][item[2]]#获取实体1的类型
        eca_2,out_2=posi(txt,em2)
        o1_type_id=opt["obj_type2id"][item[6]]
        rela=opt['predicate2id'][item[3]]

        # print("元素：{}，提取情况：{}".format(item,tokenizer.decode(txt[eca_1:out_1+1])))
        # print("元素：{}，提取情况：{}".format(item,txt[eca_2:out_2+1]))
# #         duibi()
#         out_1 = eca_1+len(j[0])
#         eca_2=txt.index(em2[1])
#         out_2 = eca_2+len(j[1])
        if not( txt[eca_1:out_1]==em1[1:-1] and em2[1:-1]==txt[eca_2:out_2]):
            print("errors 提取错误")
        listone.append({"{}".format(e1_type_id):[eca_1,out_1+1],"{}".format(o1_type_id):[eca_2,out_2+1],"relaid":rela})
        # listone.append()
#         print((tokenizer.encode(item["em1Text"]),tokenizer.encode(item["em2Text"])))
    # for i in listone:。
    # print(type(listone))
    # print(count)
    return listone
# listx=['data\\WebNLG\\clean_WebNLG\\new_test.json',""]


def sch(args):
    import codecs
    print("building schemas...")
    all_schemas = set()
    subj_type  = set()
    obj_type = set()
    # min_count = 2
    # pos_tags = set()
    # chars = defaultdict(int)
    for files in args["train_file"]:
        with open(files,'r',encoding="utf-8") as f:
            a = json.load(f)
            for ins in a:
                for spo in ins['spo_details']:
                    all_schemas.add(spo[3])
                    subj_type.add(spo[2])
                    obj_type.add(spo[6])
                # for pos in ins['pos_tags']:
                #     pos_tags.add(pos)
                # for token in ins['tokens']:
                #     for char in token:
                #         chars[char] += 1
        id2predicate = {i+1:j for i,j in enumerate(all_schemas)} # 0表示终止类别
        predicate2id = {j:i for i,j in id2predicate.items()}

        id2subj_type = {i+1:j for i,j in enumerate(subj_type)} # 0表示终止类别
        subj_type2id = {j:i for i,j in id2subj_type.items()}

        id2obj_type = {i+1:j for i,j in enumerate(obj_type)} # 0表示终止类别
        obj_type2id = {j:i for i,j in id2obj_type.items()}

    with codecs.open(args["schema_file"], 'w', encoding='utf-8') as f:
        json.dump([id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id], f, indent=4, ensure_ascii=False)
    return id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id
# 使用bert分词器
def build_data(args,tokenizer):
    import pickle
    '''
    args
        dir
        tokenmodel name
        dir-->schemas.json
    '''
    opt={}
    # os.chdir(args['dir'])
    # print(args['dir'])
    dirx=os.listdir(args['dir'])
    filename=args['dir'].split("\\")[-2]
    # from transformers import BertTokenizer
    if args['load_method']=="local":
        ModelName=args['LocalModel']
        faModel=ModelName.split('/')[-1]

    else:
        ModelName=args['TokenModel']
        faModel=ModelName.split("_")[0]
    # tokenizer=BertTokenizer.from_pretrained(ModelName)
    schefil=args['dir'] + '/schemas.json'
    if os.path.exists(schefil)and not(args['resume']):
        id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id = json.load(open(args['data_dir'] + '/schemas.json')) #获得类型id与标签映射
    else:
        x={"train_file":[args["dir"]+"/train.json",args["dir"]+"/dev.json",args["dir"]+"/test.json"],"schema_file":schefil}
        id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id=sch(x)

    opt["id2predicate"]=id2predicate
    opt['predicate2id']=predicate2id
    opt['id2subj_type']=id2subj_type
    opt['subj_type2id']=subj_type2id
    opt['id2obj_type']=id2obj_type
    opt['obj_type2id']=obj_type2id
    if not args['combine']:
        for files in dirx:
            if files.split('.')[-2]in['train','test','dev']:
                print(files)
                with open(os.path.join(args['dir'],files),'r',encoding="utf-8")as f1:
                    with open(args['dir']+"/{}_{}.json".format(filename,faModel),"ab+")as f:
                        listx=json.load(f1)
                        for itm in listx:
                            # dictx={"context":" ".join(itm['tokens']),"triple":chuli(itm,tokenizer=tokenizer)}
                            json.dump({"context":" ".join(itm['tokens']),"triple":chuli(itm,opt=opt,tokenizer=tokenizer,)},f)     
                            f.write("\n")
    else:
        # for files in dirx:
        with open(args['dir']+'/{}_{}.json'.format(filename,faModel),'ab+')as f:
            for files in dirx:
                if files.split('.')[-2] in ['train','test','dev']:
                    print(files)
                    # with open(os.path.join(args['dir'],'Pickle_{}_{}'.format(filename,ModelName.split('_')[0]),'ab+'))as f:
                    with open(os.path.join(args['dir'],files),'r',encoding="utf-8")as f1:
                        listx=json.load(f1)
                        for itm in listx:
                            
                            json.dump({"context":" ".join(itm['tokens']),"triple":chuli(itm,opt=opt,tokenizer=tokenizer)},f)
                            f.write("\n")
    return '{}_{}.json'.format(filename,faModel)


import argparse
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    os.chdir("RETask-main/RETask-main/Bert-bilstm-cnn/dataset/NYT-multi")
    parser.add_argument("--dir",type=str,default='RETask-main/RETask-main/Bert-bilstm-cnn/dataset/NYT-multi')
    parser.add_argument('--LocalModel',type=str,default="D:/workstation/NERWork/RETask/RE/bert")
    parser.add_argument('--TokenModel',type=str,default="bert-base-uncased")
    parser.add_argument('--load_method',type=str,default="local")
    parser.add_argument('--resume',type=bool,default=True)
    parser.add_argument('--combine',type=bool,default=True)
    
    args=parser.parse_args()
    opt=vars(args)
    build_data(opt)