# https://www.youtube.com/watch?v=NLrQZwnWjvI
# https://github.com/JACantoral/DL_fundamentals/blob/main/fundamentals_DL_CNN_cifar10_part1.ipynb
# https://www.youtube.com/watch?v=jevIdbQ_OVw


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Descargar datos
DATA_PATH = '/Users/oscarr/Development/Datasets/CIFAR10/'
NUM_TRAIN = 50000
NUM_VAL = 5000
NUM_TEST = 5000
MINIBATCH_SIZE = 64

transform_cifar = T.Compose([
                T.ToTensor(),
                T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
            ])

# Train dataset
cifar10_train = datasets.CIFAR10(DATA_PATH, train=True, download=True,
                             transform=transform_cifar)
train_loader = DataLoader(cifar10_train, batch_size=MINIBATCH_SIZE,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
#Validation set
cifar10_val = datasets.CIFAR10(DATA_PATH, train=False, download=True,
                           transform=transform_cifar)
val_loader = DataLoader(cifar10_val, batch_size=MINIBATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))
#Test set
cifar10_test = datasets.CIFAR10(DATA_PATH, train=False, download=True,
                            transform=transform_cifar)
test_loader = DataLoader(cifar10_test, batch_size=MINIBATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL, len(cifar10_test))))

# usar GPUs
# if torch.cuda.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')

# cuda. See this: https://www.youtube.com/watch?v=cGEIEnekmRM
device = torch.device('mps')
print(device)


# mostrar imagenes
classes = test_loader.dataset.classes
def plot_figure(image):
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.axis('off')
    plt.show()

rnd_sample_idx = np.random.randint(len(test_loader))
print(f'La imagen muestreada representa un: {classes[test_loader.dataset[rnd_sample_idx][1]]}')
image = test_loader.dataset[rnd_sample_idx][0]
image = (image - image.min()) / (image.max() -image.min() )
# plot_figure(image)


def plot_cifar10_grid():
    classes = test_loader.dataset.classes
    total_samples = 8
    plt.figure(figsize=(15, 15))
    for label, sample in enumerate(classes):
        class_idxs = np.flatnonzero(label == np.array(test_loader.dataset.targets))
        sample_idxs = np.random.choice(class_idxs, total_samples, replace=False)
        for i, idx in enumerate(sample_idxs):
            plt_idx = i * len(classes) + label + 1
            plt.subplot(total_samples, len(classes), plt_idx)
            plt.imshow(test_loader.dataset.data[idx])
            plt.axis('off')

            if i == 0: plt.title(sample)
    plt.show()


# plot_cifar10_grid()


# calcular accuracy
for i, (x,y) in enumerate(test_loader):
    print(i, x.shape, y.shape)


def accuracy(model, loader):
    num_correct = 0
    num_total = 0
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        for xi, yi in loader:
            xi = xi.to(device=device, dtype = torch.float32)
            yi = yi.to(device=device, dtype = torch.long)
            scores = model(xi)
            _, pred = scores.max(dim=1)
            num_correct += (pred == yi).sum()
            num_total += pred.size(0)
        return float(num_correct)/num_total


# loop de entrenamiento
def train(model, optimiser, epochs=100):
    model = model.to(device=device)
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(train_loader):
            model.train()
            xi = xi.to(device=device, dtype=torch.float32)
            yi = yi.to(device=device, dtype=torch.long)
            scores = model(xi)
            cost = F.cross_entropy(input= scores, target=yi)
            optimiser.zero_grad()
            cost.backward()
            optimiser.step()
        acc = accuracy(model, val_loader)
        print(f'Epoch: {epoch}, costo: {cost.item()}, accuracy: {acc},')


conv_k_3 = lambda channel1, channel2: nn.Conv2d(channel1, channel2, kernel_size=3, padding=1)


class CNN_class4(nn.Module):
    def __init__(self, in_channel, channel1, channel2):
        super().__init__()
        self.conv1 = conv_k_3(in_channel, channel1)
        self.bn1 = nn.BatchNorm2d(channel1)

        self.conv2 = conv_k_3(channel1, channel2)
        self.bn2 = nn.BatchNorm2d(channel2)

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        return x

channel1 = 16
channel2 = 32
channel3 = 64
channel4 = 128
epochs = 10
lr = 0.001
modelCNN5 = nn.Sequential(CNN_class4(3,channel1, channel2),
                          CNN_class4(channel2, channel4, channel4),
                          nn.Flatten(),
                          nn.Linear(in_features=8*8*channel4, out_features=512),
                          nn.ReLU(),
                          nn.Linear(in_features=512, out_features=128),
                          nn.ReLU(),
                          nn.Linear(in_features=128, out_features=64),
                          nn.ReLU(),
                          nn.Linear(in_features=64, out_features=10)
                          )

# modelCNN5 = nn.Sequential(nn.Conv2d(3, 6, 3),
#                           nn.ReLU(),
#                           nn.MaxPool2d(2),
#                           nn.Conv2d(6, 16, 3),
#                           nn.ReLU(),
#                           nn.MaxPool2d(2),
#                           nn.Flatten(),
#                           nn.Linear(in_features=16*3*3, out_features=512),
#                           nn.ReLU(),
#                           nn.Linear(in_features=512, out_features=128),
#                           nn.ReLU(),
#                           nn.Linear(in_features=128, out_features=64),
#                           nn.ReLU(),
#                           nn.Linear(in_features=64, out_features=10)
#                           )



optimiser = torch.optim.Adam(modelCNN5.parameters(), lr)

train(modelCNN5, optimiser, epochs)