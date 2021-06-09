deepl | Deep Learning from Scratch
======================
A toy Deep Learning library based on Numpy

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
    │   ├── .SGD                    # stochastic gradient descent
    │   ├── .Momentum               # stochastic gradient descent with momentum
    │   ├── .AdaGram                # adaptive gradient descent
    │   ├── .RMSprop                # root mean squared propogation
    │   └── .Adam                   # adaptive momentum
    │
    └── README.md
```
### Install

```
!pip install git+https://github.com/akanametov/deepl.git#egg=deepl
```

### Usage

```python
from deepl import nn
from deepl.nn import functional as F
from deepl.criterion import BCELoss
from deepl.optimizer import SGD
from deepl.data import Dataset, DataLoader

# define network architecture
class CNN(nn.Network):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=(7,7), stride=(2,2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(5,5), stride=(2,2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(6, 8, kernel_size=(3,3), stride=(2,2))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8*6*6, 32)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(32, 2)
        self.sigm = nn.Sigmoid()
        
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
        fx = self.sigm(fx)
        
        return fx
    
# define your dataset
class DogCatSet(Dataset):
    def __init__(self, x, y, mode='train'):
        super().__init__()
        self.X = x
        self.y = y
        self.mode=mode
        
    def __len__(self,):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.mode=='test':
            return x
        y = self.y[idx].astype(int)
        return x, y
    
# load your data as np.array 
x = np.array([...]) # np.array of size (N x C x H x W) in this case
y = np.array([...]) # np.array of size (N,)

# define loader with batch_size
dataset = DogCatSet(x, y)
dataloader = DataLoader(dataset, batch_size=50)

# assign model, critreion and optimizer
model = CNN()
criterion = BCELoss()
optimizer = Adam(lr=0.0005)

epochs=20
history={'loss':[], 'acc':[]}

# Train loop
for epoch in range(epochs):
    e_loss, e_acc = 0., 0.
    for (x, y) in dataloader:
        p = model(x).flatten()                   # obtain prediction
        loss = criterion(p, y)                   # calculate loss 
        acc = accuracy((p > 0.5).astype(int), y) # calculate accuracy
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
