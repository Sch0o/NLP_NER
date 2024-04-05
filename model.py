# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
import torch
import torch.nn as nn
from transformers import AdamW
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from torchcrf import CRF


def read_data(file):  # 读BMES
    with open(file, "r", encoding='utf-8') as f:
        all_data = f.read().split("\n")
        all_text = []
        all_label = []
        text = []
        label = []
        for data in all_data:
            if data == "":
                if text == []:
                    continue
                all_text.append(text)
                all_label.append(label)
                text = []
                label = []
            else:
                t, l = data.split(" ")
                text.append(t)
                label.append(l)
        return all_text, all_label


def build_label(train_label):  # 所有标签
    label_2_index = {"PAD": 0, "UNK": 1}
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    return label_2_index, list(label_2_index)


class BertDataset(Dataset):
    def __init__(self, all_text, all_label, label_2_index, tokenizer, max_len,is_test=True):
        self.all_text = all_text
        self.all_label = all_label
        self.label_2_index = label_2_index
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test=is_test

    def __getitem__(self, index):
        if self.is_test:
            self.max_len=len(self.all_text[index])
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]

        text_index = self.tokenizer.encode(text, add_special_token=True, max_length=self.max_len + 2,
                                           padding="max_length",
                                           truncation=True, return_tensor="pt")
        label_index = [0] + [self.label_2_index.get(l, 1) for l in label] + [0] + [0] * (self.max_len - len(label))

        text_index = torch.tensor(text_index)
        label_index = torch.tensor(label_index)

        return text_index.reshape(-1), label_index, len(label)

    def __len__(self):
        return self.all_text.__len__()


class BertNerModel(nn.Module):
    def __init__(self, lstm_hidden, class_num):
        super().__init__()

        self.bert = BertModel.from_pretrained("../bert_base_chinese")
        for name, param in self.bert.named_parameters():
            param.require_grad = False

        self.lstm = nn.LSTM(768, lstm_hidden, batch_first=True, num_layers=1, bidirectional=False)
        self.classifier = nn.Linear(lstm_hidden, class_num)

        self.crf = CRF(class_num, batch_first=True)
        # self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_text_index, batch_label=None):
        bert_out = self.bert(batch_text_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # 字符级别 篇章级别

        lstm_out, _ = self.lstm(bert_out0)

        pre = self.classifier(lstm_out)

        if batch_label is not None:
            # loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
            loss = -self.crf(pre, batch_label)
            return loss
        else:
            pre=self.crf.decode(pre)
            return pre


if __name__ == "__main__":
    # 读BMES
    train_text, train_label = read_data("data/train_bieos.txt")
    dev_text, dev_label = read_data("data/dev_bieos.txt")
    test_text, test_label = read_data("data/test_bieos.txt")
    label_2_index, index_2_label = build_label(train_label)

    tokenizer = BertTokenizer.from_pretrained("../bert_base_chinese")

    batch_size = 16
    epoch = 1
    max_len = 30
    lr = 0.00001
    lstm_hidden = 128
    num = int(len(train_label) / batch_size)
    count = 1

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BertDataset(train_text, train_label, label_2_index, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BertDataset(dev_text, dev_label, label_2_index, tokenizer, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    test_dataset=BertDataset(test_text,test_label,label_2_index,max_len,tokenizer)
    test_dataloade=DataLoader(test_dataset,batch_size=1,shuffle=False)

    model = BertNerModel(lstm_hidden, len(label_2_index)).to(device)
    opt = AdamW(model.parameters(), lr)

    best_score=-1
    for e in range(epoch):
        print('轮次：', e, "/", epoch)
        count = 0
        model.train()
        for batch_text_index, batch_label_index, label_len in train_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            loss = model.forward(batch_text_index, batch_label_index)
            loss.backward()

            opt.step()
            opt.zero_grad()
            print('batch: ', count, '/', num, f'loss:{loss:.2f}')
            # if count==100:
            #     break
            count += 1
            break

        model.eval()
        all_pre = []
        all_tag = []
        for batch_text_index, batch_label_index, label_len in dev_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model.forward(batch_text_index)

            # pre = pre.cpu().numpy().tolist()
            tag = batch_label_index.cpu().numpy().tolist()

            for p, t, l in zip(pre, tag, label_len):
                p = p[1:1 + l]
                t = t[1:1 + l]

                p = [index_2_label[i] for i in p]
                t = [index_2_label[i] for i in t]

                all_pre.append(p)
                all_tag.append(t)
        score = f1_score(all_tag, all_pre)
        if score>best_score:
            torch.save(model,"best_model.pt")
            best_score=score

        print(f"best_score:{best_score:.2f},score:{score:.2f}")
