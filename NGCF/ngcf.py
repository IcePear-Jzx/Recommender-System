import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy import sparse
import random
import time


class GNNLayer(nn.Module):
    def __init__(self, inF, outF, train_dataset):
        super().__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.inF = inF
        self.outF = outF
        self.linear = nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = nn.Linear(in_features=inF,out_features=outF)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.interActTransform.weight, std=0.01)

        data = []
        row = []
        col = []
        for u in range(self.user_num):
            for i in train_dataset.data[u]:
                data.append(1)
                row.append(u)
                col.append(i)

        self.R = sparse.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))
        empty1 = sparse.coo_matrix((self.user_num, self.user_num))
        empty2 = sparse.coo_matrix((self.item_num, self.item_num))
        A_upper = sparse.hstack([empty1, self.R])
        A_lower = sparse.hstack([self.R.transpose(), empty2])
        self.A = sparse.vstack([A_upper, A_lower])
        sumArr = self.A.sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        self.sqrt_D = sparse.diags(diag)
        self.L = self.sqrt_D * self.A * self.sqrt_D
        self.L = sparse.coo_matrix(self.L)
        I = sparse.eye(self.user_num + self.item_num)
        self.L1 = sparse.coo_matrix(self.L + I)
        self.L2 = sparse.coo_matrix(self.L)
        row = self.L1.row
        col = self.L1.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(self.L1.data)
        self.L1 = torch.sparse.FloatTensor(i, data).cuda()
        row = self.L2.row
        col = self.L2.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(self.L2.data)
        self.L2 = torch.sparse.FloatTensor(i, data).cuda()

    def forward(self, embed):
        # t1 = time.time()
        embed2 = torch.mul(embed, embed)
        # t2 = time.time()
        inter_part1 = self.linear(torch.sparse.mm(self.L1, embed))
        # t3 = time.time()
        inter_part2 = self.interActTransform(torch.sparse.mm(self.L2, embed2))
        # t4 = time.time()
        # print(t2 - t1, t3 - t2, t4 - t3)
        return inter_part1 + inter_part2


class NGCF(nn.Module):
    def __init__(self, train_dataset):
        super().__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.factor_num = 64
        self.reg = 0

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num, self.factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.GNNLayer1 = GNNLayer(self.factor_num, self.factor_num, train_dataset)
        self.GNNLayer2 = GNNLayer(self.factor_num, self.factor_num, train_dataset)

        self.all_user = torch.LongTensor([i for i in range(self.user_num)]).cuda()
        self.all_item = torch.LongTensor([i for i in range(self.item_num)]).cuda()

    def forward(self, user, item_i, item_j):
        # t1 = time.time()
        all_user_embed = self.embed_user(self.all_user)
        all_item_embed = self.embed_item(self.all_item)
        embed = torch.cat([all_user_embed, all_item_embed], dim=0)
        # t2 = time.time()
        g_embed_1 = self.GNNLayer1(embed)
        g_embed_1 = nn.ReLU()(g_embed_1)
        g_embed_2 = self.GNNLayer2(g_embed_1)
        g_embed_2 = nn.ReLU()(g_embed_2)
        embed = torch.cat([embed.clone(), g_embed_1.clone(), g_embed_2], dim=1)
        # t3 = time.time()
        # print(t2 - t1, t3 - t2)

        user = embed[user]
        item_i = embed[self.user_num + item_i]
        item_j = embed[self.user_num + item_j]

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        regularizer = self.reg * (torch.sum(user ** 2) + 
            torch.sum(item_i ** 2) + torch.sum(item_j ** 2))
        loss = regularizer - (prediction_i - prediction_j).sigmoid().log().sum()
        return prediction_i, prediction_j, loss


class NGCFData(data.Dataset):
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
            item_j = random.randint(0, self.item_num - 1)
            while item_j in self.data[user_id]:
                item_j = random.randint(0, self.item_num - 1)
            self.samples.append((user_id, item_i, item_j))


def get_recall(model, train_dataset, test_dataset, top_k):
    item = torch.LongTensor(np.array(range(91599)))

    users = []
    items = []
    for user_id in range(50):
        user = torch.ones(91599, dtype=torch.int64) * user_id
        users.append(user)
        items.append(item)
    
    users = torch.cat(users).cuda()
    items = torch.cat(items).cuda()
    predictions, _, _ = model(users, items, items)

    hit_count = 0
    total_count = 0
    for user_id in range(50):
        item_set = test_dataset.data[user_id]
        prediction = predictions[user_id * 91599 : (user_id + 1) * 91599].cpu()
        _, indices = torch.topk(prediction, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        train_positive = set(train_dataset.data[user_id]) & set(recommends)
        while len(train_positive) + top_k > len(recommends):
            _, indices = torch.topk(prediction, top_k + len(train_positive))
            recommends = torch.take(item, indices).cpu().numpy().tolist()
            train_positive = set(train_dataset.data[user_id]) & set(recommends)

        hit_count += len(set(item_set) & (set(recommends) - train_positive))
        total_count += len(item_set)

    return hit_count / total_count


if __name__ == '__main__':

    train_dataset = NGCFData(path='Amazon-Book/train.txt', training=True)
    test_dataset = NGCFData(path='Amazon-Book/test.txt', training=False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=train_dataset.user_num, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=test_dataset.user_num, shuffle=False, num_workers=0)

    model = NGCF(train_dataset)
    model.cuda()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10001):
        model.train()
        # t1 = time.time()
        train_loader.dataset.gen_samples()
        # t2 = time.time()

        for user, item_i, item_j in train_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()
            model.zero_grad()
            prediction_i, prediction_j, loss = model(user, item_i, item_j)
            loss.backward()
            optimizer.step()
        # t3 = time.time()
        # print(t2-t1, t3-t2)
        if epoch % 20 == 0:
            model.eval()
            recall = get_recall(model, train_dataset, test_dataset, 20)
            print('Epoch{} Recall@20: {}'.format(epoch, recall))
        else:
            print('Epoch{}'.format(epoch))
