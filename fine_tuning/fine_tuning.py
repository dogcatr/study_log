from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from typing import OrderedDict
import os


# https://pystyle.info/pytorch-list-of-transforms/
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),  # ランダムに切り抜いたあとにリサイズを行う
    transforms.RandomHorizontalFlip(),  # ランダムに左右反転を行う
    transforms.ToTensor(),  # PIL Image オブジェクトをテンソルに変換し、値の範囲を [0, 255] から [0, 1] にスケールする
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化を行う RGB毎の平均・標準偏差
])
data_dir = 'data/hymenoptera_data'
image_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    data_transform  # データの前処理
)
dataloader = torch.utils.data.DataLoader(  # バッチの設定
    image_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=10
)
dataset_size = len(image_dataset)  # データの個数
class_names = image_dataset.classes  # trainディレクトリ下にants, beesフォルダがあるから、それが名前に
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    with tqdm(range(num_epochs)) as progressbar:
        for epoch in progressbar:
            model.train()
            running_corrects = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / dataset_size
            scheduler.step()

            progressbar.set_postfix(
                OrderedDict(
                    Accuracy=epoch_acc.item(),
                )
            )


# (512->1000)のレイヤーを(512->2)のレイヤーに置き換え
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# print(model_ft.fc)
model_ft.fc = nn.Linear(num_ftrs, 2)
# print(model_ft.fc)
model_ft = model_ft.to(device)

# 最適化に使用するモジュール
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# ファインチューニング
model_ft = train_model(
    model=model_ft,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    num_epochs=25
)
