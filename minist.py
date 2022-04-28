#!/usr/bin/python3
# encoding:utf-8

import torch
from torch import nn

import torchvision
from torchvision import transforms
from torchkeras import summary
from sklearn.metrics import accuracy_score
import datetime
import numpy as np
import pandas as pd

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class NetCov(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout2d(p=0.5),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NetFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.5),
            nn.Linear(64, 10)]
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true.cpu(), y_pred_cls.cpu())


def train_step(model, features, labels):
    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()


@torch.no_grad()
def valid_step(model, features, labels):
    # 预测模式，dropout层不发生作用
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()


def train_model(model, epochs, dl_train, dl_valid, log_step_freq):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=["epoch","loss",metric_name,"val_loss","val_"+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)

    for epoch in range(1,epochs+1):

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features,labels) in enumerate(dl_train, 1):

            features = features.type(FloatTensor)
            labels = labels.type(LongTensor)
            loss,metric = train_step(model,features,labels)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:
                print(("[step = %d] loss: %.5f, "+metric_name+": %.5f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):
            features = features.type(FloatTensor)
            labels = labels.type(LongTensor)
            val_loss,val_metric = valid_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step,
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.5f,"+ metric_name +
              "  = %.5f, val_loss = %.5f, "+"val_"+ metric_name+" = %.5f")
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    return dfhistory


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(root="./data/minist/", train=True, download=True, transform=transform)
    ds_valid = torchvision.datasets.MNIST(root="./data/minist/", train=False, download=True, transform=transform)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=1024, shuffle=True, num_workers=8)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=8)

    # net = NetCov()
    net = NetFF()
    if torch.cuda.is_available():
        net.cuda()
    print(net)
    summary(net, input_shape=(1, 28, 28),input_dtype=FloatTensor)

    model = net
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        model.loss_func = nn.CrossEntropyLoss().cuda()
    else:
        model.loss_func = nn.CrossEntropyLoss()
    model.metric_func = accuracy
    model.metric_name = "accuracy"

    epochs = 10
    dfhistory = train_model(model, epochs, dl_train, dl_valid, log_step_freq=100)

