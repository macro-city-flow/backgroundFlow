import argparse
import torch.optim
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import accuracy
from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError
import torch.distributions
import numpy as np
from torch import normal

class densityForecastTask(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 pre_len: int = 1, learning_rate: float = 1e-3, feature_dim: int = 1,
                 weight_decay: float = 1.5e-3, feat_max_val: float = 1, **kwargs):
        super(densityForecastTask, self).__init__()
        self.save_hyperparameters()
        self._feature_dim = feature_dim
        self._model = model
        self._MAE = MeanAbsoluteError()
        self._transfer = nn.Sigmoid()

    def sample(self,mu: Tensor, sigma: Tensor, weights: Tensor, times: int) -> Tensor:

        # DEBUG
        # assert(mu.shape == sigma.shape == weights.shape)
        # assert(len(weights.shape) <= 2)  # no support for over 2-dimension data
        
        result = torch.FloatTensor(times, mu.shape[0])
        result = result.cuda()
        for _ in range(times):
            k = torch.multinomial(weights, num_samples=1, replacement=True).squeeze()
            
            result[_] = normal(mean=mu,std=sigma)[np.arange(mu.shape[0]), k].data
        
        return result


    def RMCI(self,mu:Tensor,sigma:Tensor,weights:torch.Tensor,y:Tensor)-> Tensor:
        
        confidence_level = 0.90#F**k.... I have no idea whether this can be used this way
        #And this can be replaced with hyperparameters 

        # DEBUG
        # assert(mu.shape == sigma.shape)
        # assert(mu.shape[0]== y.shape[0])
        # assert(len(mu.shape) <=2)#no support for over 2-dimension data
        
        samples = self.sample(mu,sigma,weights,20).transpose(0,1)#20 is a number that can be further replaced with hyperparameters
        result = [[abs(_-y[__]) for _ in samples[__]] for __ in range(samples.shape[0]) ]
        for _ in result:
            _.sort()

        result = [result[_][int(confidence_level*20)] for _ in range(len(result))]
        result = sum(result)/len(result)

        return result

    def forward(self, x):
        mu, sigma, weights = self._model(x)
        return mu, sigma, weights

    def shared_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma, weights = self(x)
        return mu, sigma, weights, self._transfer(y)

    def loss(self, mu, sigma, weights, targets):

        mixture_density = torch.distributions.Normal(loc=mu, scale=sigma)

        loss = torch.exp(mixture_density.log_prob(targets))
        loss = torch.sum(loss*weights, dim=0)
        loss = - torch.log(loss)

        return torch.mean(loss)

    def training_step(self, batch, batch_idx):

        mu, sigma, weights, y = self.shared_step(batch, batch_idx)
        loss = self.loss(mu, sigma, weights, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        mu, sigma, weights, y = self.shared_step(batch, batch_idx)

        loss = self.loss(mu, sigma, weights, y)
        mu, sigma, weights, y = mu.squeeze().transpose(0, 2).transpose(0, 1), sigma.squeeze().transpose(0, 2).transpose(
            0, 1), weights.squeeze().transpose(0, 2).transpose(0, 1), y.squeeze().transpose(0, 1)

        sample = torch.FloatTensor(y.shape)
        for _ in range(sample.shape[0]):
            sample[_] = self.sample(mu=mu[_],sigma= sigma[_],weights= weights[_],times= 1)
        sample=sample.cuda()
        mse = F.mse_loss(sample, y)
        rmse = torch.sqrt(mse)
        mae = self._MAE(sample, y)
        rmci =[self.RMCI(mu[_],sigma[_],weights[_],y[_]) for _ in range(mu.shape[0])]
        rmci = sum(rmci)/len(rmci)
        #TODO add relative interval for this
        metrics = {
            'val_loss': loss,
            'RMSE': rmse,
            'MSE': mse,
            'MAE': mae,
            'RMCI':rmci
        }

        self.log_dict(metrics)
        return mu, sigma, weights, y

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--learning-rate', '--lr',
                            type=float, default=1e-3)
        parser.add_argument('--weight-decay', '--wd',
                            type=float, default=1.5e-3)
        return parser

    @property
    def hyperparameters(self):
        return {
            'learning_rate': self._learning_rate,
            'weight_decay': self._weight_decay
        }
