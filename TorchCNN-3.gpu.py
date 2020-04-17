#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import torch
import torchvision
from torch import nn, optim
from visdom import Visdom
from tqdm.notebook import tqdm


# ## Setup dataloaders

# In[2]:


tr = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

mnist_data = torchvision.datasets.MNIST("mnist_data", download=True, transform=tr)
test_data = torchvision.datasets.MNIST("test_data", download=True, train=False, transform=tr)

dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


# ## Define CNN class

# In[3]:


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        self.linear1 = nn.Linear(3136, 256)
        self.linear2 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,X):
        n = X.size(0)
        
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        
        X = self.relu(self.conv3(X))
        X = self.relu(self.conv4(X))
        X = self.pool(X)
        
        X = X.view(n,-1)
        
        X = self.relu(self.linear1(X))
        X = self.softmax(self.linear2(X))
        return X
        


# ## Train model (took me >1hr on a laptop)
# ### If you don't want to train it, you can load it below 

# In[4]:


model = CNN1()
device=torch.device("cuda:0")
model.cuda()
loss_fn = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.SGD(params = params,lr=.01, momentum=0.9)

n_epochs = 15

for e in tqdm(range(n_epochs)):
    running_loss = 0
    for i,(images,labels) in tqdm(enumerate(dataloader)):
        #forward pass
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        
        #backward pass
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(dataloader)))


# ## Load model

# In[ ]:


model = CNN1()
model.load_state_dict(torch.load("mymodel.pt"))
model.eval()


# ## Save model

# In[ ]:


torch.save(model.state_dict(), "mymodel.pt")


# ## Test model on test set (10,000 images)

# In[ ]:


correct_count, all_count = 0, 0

incorrect = [[]]

for images,labels in testloader:
    images, labels = images.to(device), labels.to(device)
    for i in range(len(labels)):
        img = images[i].view(1, 1, 28, 28)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu().numpy()[i]
        if(true_label == pred_label):
            correct_count += 1
        else:
            incorrect.append([images[i],pred_label,true_label])
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# ### Determine the amount of incorrect predictions

# In[ ]:


incorrect = [x for x in incorrect if x != []]
len(incorrect)


# ## Visualize images that were predicted incorrectly
# #### Can be modified to show more than 5 images
# #### Arrays contain predicted label followed by actual label

# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(28,28))

for i,im in enumerate(axes.flat):
    im.imshow(incorrect[i][0].cpu().view(28,28))
    
for k in range(5):
    print(incorrect[k][1:])


# ## Views a random image that was predicted incorrectly

# In[ ]:


import random 
index = random.randrange(0,len(incorrect))
print(index)
plt.imshow(incorrect[index][0].cpu().view(28,28))
print(incorrect[index][1:])


# In[ ]:




