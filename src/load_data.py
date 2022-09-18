from sklearn.model_selection import train_test_split
import glob
import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image
import pandas as pd
import zipfile

data_path = "/home/noah/nyc-sentinel/data/2750"
def get_eurosat_dataloaders(batch_size,limit,test_size):
    
    image_df, label_dict = load_eurosat_dataset(limit=limit)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(image_df['path'], image_df['label'], test_size=test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=test_size)

    X_train.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    X_valid.reset_index(drop=True,inplace=True)
    y_valid.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)   
    
    train_dataset = EuroSATDataset(X_train,y_train,label_dict,randomize=True)
    valid_dataset = EuroSATDataset(X_valid,y_valid,label_dict,randomize=False)
    test_dataset = EuroSATDataset(X_test,y_test,label_dict,randomize=False)


    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True,num_workers = 4)
    valid_loader = DataLoader(valid_dataset,batch_size = 4, num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size = 4, num_workers=4)

    return train_loader, valid_loader, test_loader, label_dict

class EuroSATDataset(Dataset):
    def __init__(self,X,y,label_dict,randomize=True):
    
        self.X = X
        self.y = y
        self.label_dict = label_dict
        self.randomize = randomize
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        
        if self.randomize:
            preprocess = transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip()])
        else:
            preprocess = transforms.Compose([transforms.ToTensor()])        
        
        img = self.X[idx]
        x = Image.open(img)
        x = preprocess(x)
        x = (x - x.min()) / (x.max() - x.min())
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


def download_eurosat_data():
    
    os.system("wget http://madm.dfki.de/files/sentinel/EuroSAT.zip")
    os.system('mv EuroSAT.zip data/')
    zip_file = "data/EuroSAT.zip"

    try: 
        with zipfile.ZipFile(zip_file) as z:
            z.extractall()  
            os.system('mv 2750 data/')
            os.system('rm data/EuroSAT.zip')
        print("Extracted all")
    except:
        print("Invalid file")    
            
            
    return