# coding=utf-8
from model import *

train_text, train_label = read_data("data/train_bieos.txt")
test_text, test_label = read_data("data/test_bieos.txt")
label_2_index, index_2_label = build_label(train_label)
tokenizer = BertTokenizer.from_pretrained("../bert_base_chinese")

test_dataset = BertDataset(test_text, test_label, label_2_index, tokenizer, 0)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = torch.load("best_model.pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

all_pre = []
all_tag = []
test_out = []
do_input = False
if do_input:
    text = input("输入: ")
    text_index = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    text_index = text_index.to(device)

    pre = model.forward(text_index)
    pre = pre[0][1:-1]
    pre = [index_2_label[i] for i in pre]
    print(" ".join(f"{w}:{t}" for w, t in zip(text, pre)))
else:
    for id, (batch_text_index, batch_label_index, label_len) in enumerate(test_dataloader):
        text = test_text[id]

        batch_text_index = batch_text_index.to(device)
        batch_label_index = batch_label_index.to(device)
        pre = model.forward(batch_text_index)

        tag = batch_label_index.cpu().numpy().tolist()

        p = pre[0][1:-1]
        t = tag[0][1:-1]

        p = [index_2_label[i] for i in p]
        t = [index_2_label[i] for i in t]

        all_pre.append(p)
        all_tag.append(t)

        test_out.extend([f"{w} {t}" for w, t in zip(text, p)])
        test_out.append("")

        with open("test_out.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(test_out))
    score = f1_score(all_tag, all_pre)
    print(f"score:{score:.2f}")
