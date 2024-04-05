# coding=utf-8
import re

import pandas as pd

class_label = [['B-SIN', 'I-SIN', 'E-SIN', 'S-SIN'], ['B-NAME', 'I-NAME', 'E-NAME', 'S-NAME']]


def add_label(new_sents, label, site, class_num):
    for i in range(len(new_sents)):
        for n in site:
            if n == '': continue
            pos = find_all_sybstrings(new_sents[i], n)
            for p in pos:
                if len(n) == 1:
                    label[i][p] = class_label[class_num][3]
                else:
                    label[i][p:p + len(n)] = [class_label[class_num][1]] * len(n)
                    label[i][p] = class_label[class_num][0]
                    label[i][p + len(n) - 1] = class_label[class_num][2]
    return label


def split_excel():
    df = pd.read_excel('../wenshu.xls')

    shuffled_df = df.sample(frac=1)

    total_rows = len(df)
    p1_rows = int(total_rows * 0.6)
    p2_rows = int(total_rows * 0.8)
    p3_rows = total_rows

    df_train = shuffled_df.iloc[0:p1_rows]
    df_dev = shuffled_df.iloc[p1_rows:p2_rows]
    df_test = shuffled_df.iloc[p2_rows:p3_rows]
    df_train = df_train.loc[:, ['案由', '当事人', '正文']]
    df_dev = df_dev.loc[:, ['案由', '当事人', '正文']]
    df_test = df_test.loc[:, ['案由', '当事人', '正文']]

    df_train.to_excel('data/train.xlsx', index='False')
    df_test.to_excel('data/test.xlsx', index='False')
    df_dev.to_excel('data/dev.xlsx', index='False')


def find_all_sybstrings(string, sub):
    all_position = []
    position = string.find(sub)
    while position != -1:
        all_position.append(position)
        position = string.find(sub, position + 1)
    return all_position


def split_text(text):  # 文本分句
    sentences = re.split('(。|！|\!|\.|？|\?|\n|\u3000|;|；)', text)
    new_sents = []
    label = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        if sent[-1] == '\n' or '\u3000':
            sent = sent[:-1]
        if sent != "":
            new_sents.append(sent)
            label.append(['O'] * len(sent))
    return new_sents, label


def to_sentence(filename):
    all_text = []
    all_label = []
    df = pd.read_excel(filename)
    df = df.astype(str)
    for i in range(len(df)):
        if df.iloc[i, 1] != "":
            df.iloc[i, 1] = df.iloc[i, 1][2:-2]
            sin = df.iloc[i, 1].split('、')

        if df.iloc[i, 2] != "":
            name = df.iloc[i, 2]
            df.iloc[i, 2] = name[name.find('@') + 1:]
            name = df.iloc[i, 2].split(';')
        text = df.iloc[i, 3].replace(" ", "")

        new_sents, label = split_text(text)
        label = add_label(new_sents, label, sin, 0)
        label = add_label(new_sents, label, name, 1)
        all_text += new_sents
        all_label += label
    return all_text, all_label


def to_bmes(all_text, all_label, filename):
    text_out = []
    for i in range(len(all_text)):
        text_out.extend([f"{w} {t}" for w, t in zip(all_text[i], all_label[i])])
        text_out.append('')
        with open(filename, "w", encoding='utf-8') as f:
            f.write("\n".join(text_out))
        print(i, "/", len(all_text))
        if(i>10000):
            break


if __name__ == "__main__":
    is_split = False
    if is_split:
        split_excel()

    # all_text, all_label = to_sentence('data/train.xlsx')
    # to_bmes(all_text, all_label, 'data/train_bieos.txt')
    # all_text, all_label = to_sentence('data/dev.xlsx')
    # to_bmes(all_text, all_label, 'data/dev_bieos.txt')
    # all_text, all_label = to_sentence('data/test.xlsx')
    # to_bmes(all_text, all_label, 'data/test_bieos.txt')
