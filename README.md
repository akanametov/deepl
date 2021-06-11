deepl | Deep Learning from Scratch
======================
A Deep Learning library based on Numpy to represent some basic ideas in DL. 

```
deepl                 
    ├── .criterion             # Loss functions/criterions
    │   ├── .Criterion            # base Class for loss functions 
    │   ├── .BCELoss              # binary cross-entropy loss (binary class) (.Criterion)
    │   ├── .BCEWithLogitsLoss    # binary cross-entropy loss with logits (.Criterion)
    │   ├── .CrossEntropyLoss     # cross-entropy loss (.Criterion)
    │   └── . ...
    │
    ├── .data                  # Data utils
    │   ├── .Dataset              # base Class for datasets (.Sequence)
    │   ├── .DataLoader           # minibatch loader for .Dataset (object)                    
    │   └── utils                 # .Dataset and .DataLoader can also be imported from here
    │
    ├── .metrics               # Metrics
    │   ├── .accuracy             # accuracy score (function, classification)
    │   ├── .precision            # precision score (function, classification)
    │   ├── .recall               # recall score (function, classification)
    │   ├── .f1_score             # f1 score (function, classification)
    │   ├── .iou                  # intersection over union score (function, segmentation)
    │   ├── .dice                 # dice score (function, segmentation)
    │   └── .tversky              # tversky score (function, segmentation)
    │
    ├── .nn                    # Neural Network layers
    │   ├── .Parameter            # base Class for parameters    
    │   ├── .Module               # base Class for layers
    │   ├── .Network              # base Class for sequential Neural Networks
    │   ├── .Conv2d               # convolutional layer
    │   ├── .MaxPool2d            # maximum pooling layer
    │   ├── .Flatten              # flatten layer 
    │   ├── .Linear               # linear(dense, fully-connected) layer
    │   ├── .ReLU                 # relu activation layer
    │   ├── .LeakyReLU            # leaky relu activation layer
    │   ├── .Sigmoid              # sigmoid activation layer
    │   ├── .Tanh                 # hyperbolic tangence activation layer
    │   ├── .layer                # all layers can also be imported from here
    │   └── .functional        # Neural Network functions
    │       ├── .relu               # relu function
    │       ├── .leakyrelu          # leaky relu function
    │       ├── .sigmoid            # sigmoid function
    │       ├── .softmax            # softmax function
    │       ├── .tanh               # tanh function
    │       ├── .one_hot            # one-hot encoder (function)
    │       ├── .unfold             # function to unfold (N x C x H x W) tensor
    │       ├── .fold               # function to fold tensor into (N x C x H x W)
    │       └── .init          # Parameter initialization functions
    │           ├── .kaiming_uniform       # kaiming uniform initialization
    │           └── .uniform               # uniform initialization
    │
    ├── .optimizer             # Optimizers
    │   ├── .Optimizer              # base Class for optimizers
    │   ├── .SGD                    # stochastic gradient descent optimizer
    │   ├── .Momentum               # stochastic gradient descent with momentum optimizer
    │   ├── .AdaGrad                # adaptive gradient descent optimizer
    │   ├── .RMSprop                # root mean squared propogation optimizer
    │   └── .Adam                   # adaptive momentum optimizer
    │
    └── README.md
```
### Install

```
pip install git+https://github.com/akanametov/deepl.git#egg=deepl
```

### Usage

Get ready dataset:
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train = pd.read_csv('sample_data/mnist_train_small.csv', header=None).sample(1000)
assert len(np.unique(train.iloc[:, 0])) == 10

x = train.drop(labels=0, axis=1).to_numpy()
y = train[0].values.copy()
```
Define model and Train
```python
from deepl import nn
from deepl.nn import functional as F
from deepl.criterion import CrossEntropyLoss
from deepl.metrics import accuracy
from deepl.optimizer import Adam
from deepl.data import Dataset, DataLoader

# define network architecture
class CNN(nn.Network):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3,3), stride=(2,2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(3,3), stride=(2,2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(6, 8, kernel_size=(3,3), stride=(2,2))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8*2*2, 16)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x): # define forward path
        
        fx = self.conv1(x)
        fx = self.relu1(fx)
        fx = self.conv2(fx)
        fx = self.relu2(fx)
        fx = self.conv3(fx)
        fx = self.relu3(fx)
        
        fx = self.flatten(fx)
        
        fx = self.linear1(fx)
        fx = self.relu4(fx)
        fx = self.fc(fx)
        
        return fx

# define your dataset
class MyDataset(Dataset):
    def __init__(self, x, y, mode='train'):
        super().__init__()
        self.x = x
        self.y = y
        self.mode=mode
        
    def __len__(self,):
        return len(self.x)
    
    def __getitem__(self, idx):
        #print(idx)
        xi = self.x[idx]#.iloc[idx].values
        #print(xi.shape)
        xi = xi.reshape(len(xi), 1, 28, 28)/255
        if self.mode=='test':
            return xi
        yi = self.y[idx].astype(int)
        return xi, yi

# define loader with batch_size
dataset = MyDataset(x, y)
dataloader = DataLoader(dataset, batch_size=50)

# assign model, critreion and optimizer
model = CNN()
criterion = CrossEntropyLoss()
optimizer = Adam(lr=0.0005)

epochs=20
history={'loss':[], 'acc':[]}

# Train loop
for epoch in range(epochs):
    e_loss, e_acc = 0., 0.
    for (x, y) in dataloader:
        p = model(x)                             # obtain prediction
        loss = criterion(p, y)                   # calculate loss 
        acc = accuracy(np.argmax(p,axis=1), y)   # calculate accuracy
        grad = criterion.backward()              # get gradient of loss function 
        model.backward(grad)                     # backward propogate of gradient through network
        params = optimizer.step(model.parameters(), model.grads()) # calculate new parameters by optimizer
        model.update(params)                     # update parameters of network
        
        e_loss += loss.item()/len(dataloader)
        e_acc += acc.item()/len(dataloader)
    print(f'Epoch {epoch+1}/{epochs} | loss: {e_loss:.4f} | acc: {e_acc:.4f}')
    history['loss'].append(e_loss)               # save loss value
    history['acc'].append(e_acc)                 # save accuracy value
```
Results:
```
Epoch 1/100 | loss: 2.3210 | acc: 0.1210
Epoch 2/100 | loss: 2.2941 | acc: 0.1850
Epoch 3/100 | loss: 2.2794 | acc: 0.1820
Epoch 4/100 | loss: 2.2638 | acc: 0.1890
Epoch 5/100 | loss: 2.2400 | acc: 0.2120
Epoch 6/100 | loss: 2.2028 | acc: 0.2340
Epoch 7/100 | loss: 2.1405 | acc: 0.3040
Epoch 8/100 | loss: 2.0429 | acc: 0.3500
Epoch 9/100 | loss: 1.9130 | acc: 0.3650
Epoch 10/100 | loss: 1.7878 | acc: 0.3900
...
Epoch 91/100 | loss: 0.6349 | acc: 0.8090
Epoch 92/100 | loss: 0.6329 | acc: 0.8080
Epoch 93/100 | loss: 0.6310 | acc: 0.8080
Epoch 94/100 | loss: 0.6291 | acc: 0.8090
Epoch 95/100 | loss: 0.6273 | acc: 0.8070
Epoch 96/100 | loss: 0.6255 | acc: 0.8070
Epoch 97/100 | loss: 0.6239 | acc: 0.8070
Epoch 98/100 | loss: 0.6222 | acc: 0.8070
Epoch 99/100 | loss: 0.6206 | acc: 0.8080
Epoch 100/100 | loss: 0.6191 | acc: 0.8070
```
Not bad, taking into account that this is a multiclass classification:)
See the [notebook](https://github.com/akanametov/deepl/blob/main/examples/deepl.ipynb) for details.

### License

This project is licensed under MIT.
