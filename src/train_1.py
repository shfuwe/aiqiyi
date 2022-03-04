ban=0
batch_size=4

import torch
cuda_num = str(5)
device_s = "cuda:" + cuda_num
device0 = torch.device(device_s if torch.cuda.is_available() else "cpu")

cuda_num_test = str(5)
device_s = "cuda:" + cuda_num_test
device1 = torch.device(device_s if torch.cuda.is_available() else "cpu")

# 读数据
import pandas as pd
df_train=pd.read_excel("../data/df_train.xlsx",index_col=0)
df_test=pd.read_excel("../data/df_test.xlsx",index_col=0)


import ast
for i in range(len(df_train)):
    new_list=[]
    old_list=ast.literal_eval(df_train.loc[i]['label'])
    for j in old_list:
        if j==0:
            new_list+=[0,0,0]
        if j==1:
            new_list+=[0,0,1]
        if j==2:
            new_list+=[0,1,1]
        if j==3:
            new_list+=[1,1,1]

    df_train.loc[i]['label']=new_list

for i in range(len(df_test)):
    new_list = []
    old_list = ast.literal_eval(df_test.loc[i]['label'])
    for j in old_list:
        if j == 0:
            new_list += [0, 0, 0]
        if j == 1:
            new_list += [0, 0, 1]
        if j == 2:
            new_list += [0, 1, 1]
        if j == 3:
            new_list += [1, 1, 1]

    df_test.loc[i]['label'] = new_list


from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
df_train_text=[df_train['text'][i] for i in range(len(df_train))]
df_test_text=[df_test['text'][i] for i in range(len(df_test))]


text2id_train = tokenizer(
        df_train_text, max_length=100, padding='max_length', truncation=True, return_tensors="pt"
    )
input_ids_train=text2id_train["input_ids"]
mask_train=text2id_train["attention_mask"]

text2id_test = tokenizer(
        df_test_text, max_length=100, padding='max_length', truncation=True, return_tensors="pt"
    )
input_ids_test=text2id_test["input_ids"]
mask_test=text2id_test["attention_mask"]


df_train['input_ids']=input_ids_train.tolist()
df_train['mask']=mask_train.tolist()

df_test['input_ids']=input_ids_test.tolist()
df_test['mask']=mask_test.tolist()

from torch.utils.data import Dataset
class SentimentDataset(Dataset):
    def __init__(self,df):
        self.dataset = df
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
#         print(label)
        input_ids = self.dataset.loc[idx, "input_ids"]
        mask = self.dataset.loc[idx, "mask"]
        sample = {"text": text, "label": label,"input_ids":input_ids,"mask":mask}
        # print(sample)
        return sample

#按batch_size分
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import torch

train_loader = DataLoader(
    SentimentDataset(df_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
print(train_loader)

test_loader = DataLoader(
    SentimentDataset(df_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
print(test_loader)

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F


class fn_cls(nn.Module):
    def __init__(self, device):
        super(fn_cls, self).__init__()
        self.model = AutoModel.from_pretrained("bert-base-chinese")
        self.model.to(device)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.l1 = nn.Linear(768, 18)
        self.l2 = nn.Linear(18, 6)

    #         self.l2 = nn.Linear(768, 6)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        #         print(outputs[0])torch.Size([8, 100, 768])
        #         print(outputs[1])torch.Size([8, 768])
        #         print(outputs[0][:,0,:])torch.Size([8, 768])
        #         print(outputs[1])torch.Size([8, 768])
        x = outputs[1]
        x = self.dropout(x)
        x = self.l1(x)
        x = self.sig(x)

        x0 = self.l2(x)
        return x, x0

import torch

if ban==0:
    cls = fn_cls(device0)
    cls.train()
else:
    cls=torch.load("../data/cls"+str(ban)+".model",map_location=device0)


def _18to6_(output, round_=0):  # 18tensor 变 6list
    output0 = []
    for j in output:  # 18
        i = 0
        s = ''
        list6 = []
        while (i < len(j)):
            ok = j[i] * 1 + j[i + 1] * 1 + j[i + 2] * 1
            if round_ == 1:
                list6.append(ok.round().int().tolist())
            else:
                list6.append(ok.tolist())
            i += 3
        output0.append(list6)

    return torch.tensor(output0)


def get_loss_test(output, A):  # 8*18
    cor_add = (output.round() == A).sum().item()
    sum0 = 0
    for i in range(len(output)):
        for j in range(6):
            sum0 += (output[i][j] - A[i][j]) * (output[i][j] - A[i][j])
    return sum0, cor_add


def test(device_test):
    cls.to(device_test)
    cls.eval()

    correct = 0
    total = 0
    loss_test = 0
    for batch_idx, batch in enumerate(test_loader):
        label = torch.stack(batch['label']).t().to(device_test).float()
        input_ids = torch.stack(batch['input_ids']).t().to(device_test)
        mask = torch.stack(batch['mask']).t().to(device_test)

        output18, output6 = cls(input_ids, attention_mask=mask)

        total += len(output6) * 6
        label_6 = _18to6_(label, round_=1).to(device_test)

        # print(output18,output6,label,label_6)
        loss_add, cor_add = get_loss_test(output6, label_6)
        loss_test += loss_add
        correct += cor_add

        tes_score = 1 / (1 + (loss_test / total) ** 0.5)
        acc_score = 100. * correct / total

        print('[{}/{} ({:.0f}%)]\t正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}%，score：{}'.format(
            batch_idx, len(test_loader), 100. * batch_idx / len(test_loader),
            correct, total, acc_score,
            tes_score
        ), end="\r")
    print('[{}/{} ({:.0f}%)]\t正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}%，score：{}'.format(
        batch_idx, len(test_loader), 100. * batch_idx / len(test_loader),
        correct, total, acc_score,
        tes_score
    ))
    #     cls.to(device_train)
    return tes_score, acc_score


t_score,t_acc=test(device0)