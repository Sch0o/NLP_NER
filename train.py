#encoding=utf-8
from model import *
if __name__ == "__main__":
    # 读BMES
    train_text, train_label = read_data("data/train_bieos.txt")
    dev_text, dev_label = read_data("data/dev_bieos.txt")
    test_text, test_label = read_data("data/test_bieos.txt")
    label_2_index, index_2_label = build_label(train_label)

    tokenizer = BertTokenizer.from_pretrained("../bert_base_chinese")

    batch_size = 16
    epoch = 2
    max_len = 30
    lr = 0.00001
    lstm_hidden = 128
    num = int(len(train_label) / batch_size)
    count = 1

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BertDataset(train_text, train_label, label_2_index, tokenizer, max_len,False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BertDataset(dev_text, dev_label, label_2_index, tokenizer, max_len,False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

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

        model.eval()
        all_pre = []
        all_tag = []
        for batch_text_index, batch_label_index, label_len in dev_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model.forward(batch_text_index)

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