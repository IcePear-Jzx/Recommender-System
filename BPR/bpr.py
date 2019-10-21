import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random
import os
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = 0


class BPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.factor_num = 32

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num, self.factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j


class BPRData(data.Dataset):
    def __init__(self, path, training=True):
        super().__init__()
        self.path = path
        self.training = training
        self.user_num = 52643
        self.item_num = 91599
        self.data = {}
        self.load_data()
        self.samples = []

    def __getitem__(self, index):
        user_id = self.samples[index][0]
        item_i = self.samples[index][1]
        item_j = self.samples[index][2]
        if self.training:
            return user_id, item_i, item_j
        else:
            return user_id, item_i, item_i

    def __len__(self):
        return self.user_num

    def load_data(self):
        with open(self.path, 'r') as f:
            for line in f:
                line = [int(i) for i in line.split()]
                user = line[0]
                item_set = line[1:]
                self.data[user] = item_set

    def gen_samples(self):
        self.samples = []
        for user_id in range(self.user_num):
            item_i = random.choice(self.data[user_id])
            item_j = np.random.randint(self.item_num)
            while item_j in self.data[user_id]:
                item_j = np.random.randint(self.item_num)
            self.samples.append((user_id, item_i, item_j))


def get_recall(model, test_dataset, top_k):
    hit_count = 0
    total_count = 0
    item_i = item_j = torch.LongTensor(np.array(range(91599)))
    item_i = item_i.cuda()
    item_j = item_j.cuda()
    for user_id, item_set in test_dataset.data.items():
        if user_id >= 100:
            break
        user = torch.ones(91599, dtype=torch.int64) * user_id
        user = user.cuda()

        prediction_i, prediction_j = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(item_i, indices).cpu().numpy().tolist()

        hit_count += len(set(item_set) & set(recommends))
        total_count += len(item_set)

    return hit_count / total_count


if __name__ == '__main__':

    train_dataset = BPRData(path='amazon-book/train.txt', training=True)
    test_dataset = BPRData(path='amazon-book/test.txt', training=False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=4096, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=test_dataset.user_num, shuffle=False, num_workers=0)

    model = BPR()
    model.cuda()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10001):
        model.train()
        t1 = time.time()
        train_loader.dataset.gen_samples()
        t2 = time.time()
        
        for user, item_i, item_j in train_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()
            model.zero_grad()
            prediction_i, prediction_j = model(user, item_i, item_j)
            loss = - (prediction_i - prediction_j).sigmoid().log().sum()
            loss.backward()
            optimizer.step() 
        t3 = time.time()
        print(t2-t1, t3-t2)

        if epoch % 10 == 0:
            model.eval()
            recall = get_recall(model, test_dataset, 20)
            print('Epoch{} Recall@20: {}'.format(epoch, recall))
