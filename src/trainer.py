from sklearn.metrics import accuracy_score
from load_data import get_eurosat_dataloaders
import torchvision
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser

class EuroSATTrainer(pl.LightningModule):
    def __init__(self,
                 args,
                 num_classes):
        super().__init__()
        self.save_hyperparameters() # turns args to self.hparams
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.limit = args.limit
        self.test_size = args.test_size
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(nn.Linear(512, num_classes),
                          nn.Softmax(dim=1))        
        
    def forward(self,x):

        return self.model(x)

    def training_step(self, batch, batch_idx):

        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(input=y_pred,target=y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    # validation step
    def validation_step(self,batch,batch_idx):

        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(input=y_pred,target=y)
        self.log('valid_loss', loss, on_epoch=True)
        return {'predicted': y_pred, 'truth': y, 'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        y_pred = np.array([])
        y_true = np.array([])
        for out in validation_step_outputs:

            y_pred = np.concatenate([y_pred,out['predicted'].cpu().numpy().argmax(axis=1).flatten()])
            y_true = np.concatenate([y_true,out['truth'].cpu().numpy().flatten()])
        
        acc_score = accuracy_score(y_pred,y_true)
        self.log('valid_acc', acc_score, on_epoch=True)

    
    # validation step
    def test_step(self,batch,batch_idx):

        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(input=y_pred,target=y)
        self.log('test_loss', loss, on_epoch=True)
        return {'predicted': y_pred, 'truth': y, 'loss': loss}        

    def test_epoch_end(self, test_step_outputs):
        y_pred = np.array([])
        y_true = np.array([])
        for out in test_step_outputs:

            y_pred = np.concatenate([y_pred,out['predicted'].cpu().numpy().argmax(axis=1).flatten()])
            y_true = np.concatenate([y_true,out['truth'].cpu().numpy().flatten()])
        
        acc_score = accuracy_score(y_pred,y_true)
        self.log('test_acc', acc_score, on_epoch=True)
        
    def prepare_data(self):
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        self.train_loader, self.valid_loader, self.test_loader, self.label_dict = get_eurosat_dataloaders(batch_size=self.batch_size,
                                                                                        limit = self.limit,
                                                                                        test_size = self.test_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader
    
    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr = self.learning_rate ,weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 3)
        return  {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--learning_rate', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=3e-4)
        parser.add_argument('--batch_size', type=int,
                            default=64)
        parser.add_argument('--limit', type=int,
                        default=3000)
        parser.add_argument('--test_size', type=float,
                        default=.15)
        return parser
    
    
def main():
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = EuroSATTrainer.add_model_specific_args(parser)
    args = parser.parse_args()
    

    checkpoint_callback = ModelCheckpoint(
        save_last = True,
    save_top_k=1,
    verbose=True,
    monitor="valid_loss",
    mode="min"
    )
    
    early_stopping_callback = EarlyStopping(
                       monitor='valid_loss',
                       min_delta=0.00,
                       patience=5,
                       verbose=False,
                       mode='min'
                    )

    lr_monitor = LearningRateMonitor(logging_interval= 'step')

    run = wandb.init()
    wandb_logger = WandbLogger() 

    trainer = pl.Trainer().from_argparse_args(args,logger=wandb_logger,
                       callbacks=[checkpoint_callback,lr_monitor,early_stopping_callback])

    landclassifier = EuroSATTrainer(args,num_classes=10)

    if args.auto_lr_find:
        trainer.tune(landclassifier)

    trainer.fit(landclassifier)
    trainer.test()
    wandb.finish()


if __name__ == '__main__':
    main()