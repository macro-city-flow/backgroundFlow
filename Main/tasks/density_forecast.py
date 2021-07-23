import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import utils.metrics
import utils.losses
import torch.distributions

class densityForecastTask(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 pre_len: int = 1, learning_rate: float = 1e-3, feature_dim: int =1,
                 weight_decay: float = 1.5e-3, feat_max_val:float=1, **kwargs):
        super(densityForecastTask, self).__init__()
        self.save_hyperparameters()
        self._feature_dim = feature_dim
        self._model = model

    def forward(self, x):
        mu,sigma,weights=self._model(x)
        return mu,sigma,weights

    def shared_step(self, batch, batch_idx):
        x, y = batch
        mu,sigma,weights = self(x)
        return mu,sigma,weights, y
    
    def loss(self,mu,sigma,weights, targets):
        
        mixture_density = torch.distributions.Normal(loc = mu,scale = sigma)
        
        loss = torch.exp(mixture_density.log_prob(targets))
        loss = torch.sum(loss*weights,dim = 0)
        loss = - torch.log(loss)

        return torch.mean(loss)        


    def training_step(self, batch, batch_idx):
                
        mu,sigma,weights, y = self.shared_step(batch, batch_idx)
        loss = self.loss(mu,sigma,weights, y)
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):

        mu,sigma,weights, y = self.shared_step(batch, batch_idx)
        
        loss = self.loss(mu,sigma,weights,y)
        mixture_density = torch.distributions.Normal(loc = mu,scale = sigma)
        
        prob = torch.exp(mixture_density.log_prob(y))
        sample = torch.FloatTensor(y.shape)

        #TODO multinomial only support 2-dimension so here needs modification
        #TODO I need to realize sampling for loss calculation
        #TODO I need to realize confidence interval as a metric        
        metrics = {
            'val_loss': loss
        }
        self.log_dict(metrics)
        return mu,sigma,weights,y

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),  
                                lr=self.hparams.learning_rate, 
                                weight_decay=self.hparams.weight_decay)

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3)
        parser.add_argument('--weight-decay', '--wd', type=float, default=1.5e-3)
        return parser


    @property
    def hyperparameters(self):
        return {
            'learning_rate':self._learning_rate,
            'weight_decay':self._weight_decay
        }
