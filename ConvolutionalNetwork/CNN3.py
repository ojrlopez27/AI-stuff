
# https://www.youtube.com/watch?v=jevIdbQ_OVw
import torch.optim
from torch.utils.data import Subset, DataLoader
import sklearn.model_selection
import torchvision
import torch.nn as nn

mnist_train_data = torchvision.datasets.MNIST(root='~/Development/Datasets',
                                              train=True,
                                              download=True,
                                              transform=torchvision.transforms.ToTensor())

mnist_test_data = torchvision.datasets.MNIST(root='~/Development/Datasets',
                                             train=False,
                                             download=True,
                                             transform=torchvision.transforms.ToTensor())

# Set de entrenamiento y validación estratificados
sss = sklearn.model_selection.StratifiedShuffleSplit(train_size=0.75).split(mnist_train_data.data,
                                                                            mnist_train_data.targets)
train_idx, val_idx = next(sss)

# Dataloder de entrenamiento
train_dataset = Subset(mnist_train_data, train_idx)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

# Dataloader de validación
valid_dataset = Subset(mnist_train_data, val_idx)
valid_loader = DataLoader(train_dataset, shuffle=False, batch_size=256)

class RedConvolucional(nn.Module):

    def __init__(self):
        super(type(self), self).__init__()
        # la entrada son imagenes de 1 x 32 x 32
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5, padding=1),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2),
                                      nn.Conv2d(6, 16, 5),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2))

        self.classifier = nn.Sequential(nn.Linear(16*5*5, 120),
                                        nn.ReLU(),
                                        nn.Linear(120, 84),
                                        nn.ReLU(),
                                        nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        # Esto es de tamaño Mx16x5x5
        x = x.reshape(-1, 16*5*5)
        # Esto es de tamaño Mx400
        x = self.classifier(x)
        return x



from ignite.engine import Engine, Events
from ignite.metrics import Loss, Accuracy

model = RedConvolucional()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
max_epochs = 100
device = torch.device('mps')

model = model.to(device)

# Esto es lo que hace el engine de entrenamiento
def train_one_step(engine, batch):
    optimizer.zero_grad()
    x, y = batch
    x, y = x.to(device), y.to(device)
    yhat = model.forward(x)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    return loss.item() # Este output se puede llamar luego como trainer.state.output

# Esto es lo que hace el engine de evaluación
def evaluate_one_step(engine, batch):
    with torch.no_grad():
        x, y = batch
        x, y = x.to(device), y.to(device)
        yhat = model.forward(x)
        return yhat, y

trainer = Engine(train_one_step)
evaluador = Engine(evaluate_one_step)
metrics = {'Loss': Loss(criterion), 'Acc': Accuracy()}
for name, metric in metrics.items():
    metric.attach(evaluador, name)

import time
from torch.utils.tensorboard import SummaryWriter
from ignite.handlers import ModelCheckpoint

# Contexto de escritura para datos de tensorboard
with SummaryWriter(log_dir=f'/tmp/tensorboard/lenet5_mnist_{time.time_ns()}') as writer:

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def log_results(engine):
        # Evaluo el conjunto de entrenamiento
        evaluator.run(train_loader)
        writer.add_scalar('train/loss', evaluador.state.metrics['Loss'], engine.state.epoch)
        writer.add_scalar('train/accy', evaluador.state.metrics['Acc'], engine.state.epoch)

        # Evaluo el conjunto de validación
        evaluator.run(valid_loader)
        writer.add_scalar('train/loss', evaluador.state.metrics['Loss'], engine.state.epoch)
        writer.add_scalar('train/accy', evaluador.state.metrics['Acc'], engine.state.epoch)

    # Guardo el mejor modelo en validación
    best_model_handler = ModelCheckpoint(dirname='.', require_empty=False, filename_prefix='best', n_saved=1,
                                         score_function=lambda engine: -engine.state.metrics['Loss'],
                                         score_name='val_loss')

    # Lo siguiente se ejecuta cada vez que termine el loop de validación
    evaluador.add_event_handler(Events.COMPLETED,
                                best_model_handler, {'lenet5': model})

    trainer.run(train_loader, max_epochs=max_epochs)
