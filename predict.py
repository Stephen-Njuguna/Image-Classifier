import json
import PIL
import torch
import numpy as np
from torchvision import models,transforms
import matplotlib as plt
import train


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
# open the image
    img = PIL.Image.open(image)

#Resizing an cropping the images
    preprocess_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])

# preprocess the image
    img = preprocess_image(img)

# return the preprocessed image as a numpy array
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5): 
    ''' Predict the class (or classes) of an image using a trained deep learning model. ''' 
    model.to('cpu') 
    image = process_image(image_path)
    image = image.unsqueeze_(0) 
    #image = torch.from_numpy(image) 
    image = image.float() 
    with torch.no_grad(): 
        model.to('cpu') 
        outputs = model.forward(image.cpu()) 
        probs, classes = torch.exp(outputs).topk(topk)
    return probs[0].tolist(), classes[0].add(1).tolist() 

def main():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

        image_path = 'flowers/test/100/image_07896.jpg'

        plt.figure(figsize=[5,10])

        ax = plt.subplot(2,1,1)

        # Viewing the image 

        img = imshow(process_image(image_path),ax=ax)
        ax.set_yticks([])
        ax.set_xticks([])

        #Viewing the top 5 classes 

        ax = plt.subplot(2,1,2)
        prob, clss = predict(image_path,model)

        clss_name = [cat_to_name[str(i)] for i in clss]

        plt.barh(range(len(clss_name)),prob)
        plt.yticks(range(len(clss_name)),clss_name)
        plt.gca().invert_yaxis()

    if __name__ == '__main__':
        main()
