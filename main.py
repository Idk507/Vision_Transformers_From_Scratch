import numpy as np 
import torch 
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor 
from torchvision.datasets.mnist import MNIST 
from torchvision import transforms
from torch import nn
from VIT import VIT
from MultiHeadAttention import MultiHeadAttention
from EncoderVIT import EncoderVIT
torch.manual_seed(0)



#train 

def main():
    transform = ToTensor()
    train_set = MNIST(root= './../datasets',train=True,download=True,transform=transform)
    test_set = MNIST(root= './../datasets',train=False,download=True,transform=transform)

    train_loader = DataLoader(train_set,shuffle=True,batch_size=128)
    test_loader = DataLoader(test_set,shuffle=True,batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VIT(chw=(1,28,28),n_patches=7,n_heads=2,
                 n_blocks=2,hidden_d=2,out_d=10)
    model.to(device)
    
    n_epochs = 5 
    lr = 0.001

    #training 
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    for epoch in trange(n_epochs,desc='Training'):
        train_loss = 0.0 
        for batch in tqdm(train_loader,desc=f"Epoch {epoch+1} in training"):
            x,y = batch #[0,0,1,0,....] #1,28,28
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat,y)

            train_loss += loss.detach().cpu().item()/len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss : {train_loss}")
        
        
    #tesing
    with torch.no_grad:
        correct,total = 0,0
        test_loss = 0.0
        for batch in tqdm(test_loader,dec="testing"):
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat,y)
            test_loss += loss.detach().cpu().item()/len(test_loader)
            pred = torch.argmax(y_hat,dim=1)
            correct += torch.sum(pred == y).detach().cpu().item()
            total += len(y)
        print(f"Test loss : {test_loss}")
        print(f"Accuracy : {correct/total}")
    
    #save model
    torch.save(model.state_dict(),'model.pth')