# Importing Libraries
import matplotlib.pyplot as plt 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
import time
from PIL import Image
import numpy as np
import seaborn as sns


def initialize(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
    'train_transform':transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

    'test_transforms':transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
        
    'validation_transforms':transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
        
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train_data': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train_transform']),
        'test_data': datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test_transforms']),
        'valid_data': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['validation_transforms'])}


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64)
    validloader = torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)

    return trainloader, testloader,validloader


def create_model(structure= 'vgg16',learning_rate=0.03,dropout=0.2):
    if structure == 'vgg16':
        model = models.VGG16(pretrained= True)

    else:
        model = models.densenet121(pretrained=True)


    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(1024, 102),
                                    nn.LogSoftmax(dim=1))

    model.to('cuda')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    return model,criterion,optimizer

def train_model(model,criterion,optimizer,dataloader,validloader,device ='cpu',print_every=50,epochs=10):
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = print_every
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train() 

            return model

def validation(model,dataloader,criterion,device):
    loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images,labels = images.to('cuda'), labels.to('cuda')
            
            output = model.forward(images)
            loss += criterion(output,labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim = 1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
            
    return loss, accuracy

def save_checkpoints(model,image_datasets,optimizer):
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    model.cpu
    checkpoint={'structure':'vgg16',
                'hidden_layer1':120,
                'drouput':0.5,
                'state_dict':model.state_dict(),
                'optimizer_dict':optimizer.state_dict(),
                'class_to_idx':model.class_to_idx}
                

    return torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoints(path='checkpoint.pth'):
    checkpoint = torch.laod('checkpoint.pth')
    model = models.vgg16(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idz = checkpoint['class_to_idx']
    model.load_state_dict=checkpoint['state_dict']

    return model

