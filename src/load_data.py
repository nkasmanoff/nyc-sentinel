from sklearn.model_selection import train_test_split
import glob
import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image
import pandas as pd

data_path = "/home/noah/nyc-sentinel/data/2750"
def get_eurosat_dataloaders(batch_size,limit,test_size):
    
    image_df, label_dict = load_eurosat_dataset(limit=limit)
    
    X_train, X_test, y_train, y_test = train_test_split(image_df['path'], image_df['label'], test_size=test_size)

    X_train.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)   
    
    train_dataset = EuroSATDataset(X_train,y_train,label_dict)
    valid_dataset = EuroSATDataset(X_test,y_test,label_dict)


    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True,num_workers = 4)
    valid_loader = DataLoader(train_dataset,batch_size = 4, num_workers=4)

    return train_loader, valid_loader

class EuroSATDataset(Dataset):
    def __init__(self,X,y,label_dict):
    
        self.X = X
        self.y = y
        self.label_dict = label_dict
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        
        preprocess = transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   # normalization used on subset training data
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                       std=[0.229, 0.224, 0.225])]) # better mean and std from https://www.kaggle.com/code/maunish/eurosat-pytorch-train-effecientnet/notebook
        
        img = self.X[idx]
        x = Image.open(img)
        x = preprocess(x)
        y = torch.tensor(self.y[idx])
        return x, y
    
    
def load_eurosat_dataset(limit=None):
    """
    
    Using the extracted folder for EuroSAT dataset, transforms data into training and test sets to be processed via dataloaders.
    
    Based on https://github.com/thegomeslab/dsces/blob/2e7f0a9e1b5761b78857d8ee709e6ec09421bef7/lectures/18b_Convolutional_Neural_Networks_EuroSAT.ipynb
    """
    
    
    data_folders = sorted(glob.glob(f"{data_path}/*"))
    # preprocessing steps for image

    paths = []
    labels = []
    label_dict = {}

    for idx, folder in enumerate(data_folders):
        label_dict[idx] = folder.replace(f'{data_path}','')
        imgs = sorted(glob.glob(folder + "/*.jpg"))
        if limit:
            imgs = imgs[:limit]
        for i in tqdm.tqdm(imgs):
            paths.append(i)
            labels.append(idx)
            
    image_df = pd.DataFrame()
    image_df['path'] = paths
    image_df['label'] = labels
    image_df = image_df.sample(frac = 1)
    return image_df, label_dict
