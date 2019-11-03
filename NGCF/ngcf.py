import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import random
import time


class GNNLayer(Module):
    def __init__(self,inF,outF):
        super().__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat
        L1 = L1
        inter_feature = torch.mul(features,features)

        inter_part1 = self.linear(torch.sparse.mm(L1,features))
        inter_part2 = self.interActTransform(torch.sparse.mm(L2,inter_feature))

        return inter_part1+inter_part2


class BPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.factor_num = 64
        self.reg = 0.001

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num, self.factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.GNNLayer1 = GNNLayer(R)
        self.GNNLayer2 = GNNLayer(R)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        user = self.GNNLayer1(user)
        user = self.GNNLayer2(user)
        item_i = self.embed_item(item_i)
        item_i = self.
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        regularizer = self.reg * (torch.sum(user ** 2) + 
            torch.sum(item_i ** 2) + torch.sum(item_j ** 2))
        loss = regularizer - (prediction_i - prediction_j).sigmoid().log().sum()
        return prediction_i, prediction_j, loss


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
            item_j = random.randint(0, self.item_num - 1)
            while item_j in self.data[user_id]:
                item_j = random.randint(0, self.item_num - 1)
            self.samples.append((user_id, item_i, item_j))


def get_recall(model, train_dataset, test_dataset, top_k):
    hit_count = 0
    total_count = 0
    item_i = item_j = torch.LongTensor(np.array(range(91599)))
    item_i = item_i.cuda()
    item_j = item_j.cuda()
    for user_id in range(100):
        item_set = test_dataset.data[user_id]

        user = torch.ones(91599, dtype=torch.int64) * user_id
        user = user.cuda()

        prediction_i, _, _ = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(item_i, indices).cpu().numpy().tolist()
        train_positive = set(train_dataset.data[user_id]) & set(recommends)
        while len(train_positive) + top_k > len(recommends):
            _, indices = torch.topk(prediction_i, top_k + len(train_positive))
            recommends = torch.take(item_i, indices).cpu().numpy().tolist()
            train_positive = set(train_dataset.data[user_id]) & set(recommends)

        hit_count += len(set(item_set) & (set(recommends) - train_positive))
        total_count += len(item_set)

    return hit_count / total_count


if __name__ == '__main__':

    train_dataset = BPRData(path='Amazon-Book/train.txt', training=True)
    test_dataset = BPRData(path='Amazon-Book/test.txt', training=False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=4096, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=test_dataset.user_num, shuffle=False, num_workers=0)

    model = BPR()
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

        if epoch % 10 == 0:
            model.eval()
            recall = get_recall(model, train_dataset, test_dataset, 20)
            print('Epoch{} Recall@20: {}'.format(epoch, recall))
