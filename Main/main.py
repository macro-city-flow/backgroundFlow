import argparse
from utils.callbacks import SendMessageCallback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.misc
import torch

DATA_PATHS = {
    'chengdu': {
        'feat': 'data/chengdu_data.pkl',
        'adj': 'data/chengdu_adj.pkl'
    }
}


def get_model(args, dm):
    if args.model_name == 'GCN':
        model = models.GCN(adj=dm.adj, input_dim=args.output_dim,
                           hidden_dim=args.hidden_dim, feature_dim=args.output_dim, output_dim=args.output_dim)
    if args.model_name == 'GRU':
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim,feature_dim=args.output_dim,output_dim=args.output_dim,fr =args.gamma)
    if args.model_name == 'TGCN':
        model = models.TGCN(adj=dm.adj,feature_dim=args.output_dim,input_dim = args.output_dim,hidden_dim=args.hidden_dim,output_dim=args.output_dim,fr=args.gamma)
    if args.model_name == 'MDN':
        model = models.MDN(input_dim=args.output_dim,output_dim=args.output_dim,feature_dim=args.output_dim,gamma=args.gamma)
    if args.model_name == 'GRU_MDN':
        model = models.GRU_MDN(input_dim=args.output_dim,output_dim=args.output_dim,feature_dim=args.output_dim,gamma=args.gamma)
    if args.model_name == 'GRU_MDN2':
        model = models.GRU_MDN2(input_dim=args.output_dim,output_dim=args.output_dim,feature_dim=args.output_dim,gamma=args.gamma)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings + 'Task')(model=model,
                                                  feat_max_val=dm.feat_max_val,
                                                  **vars(args))
    return task


def get_callbacks():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='train_loss')
    print_best_epoch_metrics_callback = utils.callbacks.PrintBestEpochMetricsCallback(
        monitor='train_loss')
    send_message_callback = SendMessageCallback(monitor='train_loss')
    callbacks = [checkpoint_callback, print_best_epoch_metrics_callback,send_message_callback]
    return callbacks


def main_forecast(args):
    dm = getattr(utils.data, temp_args.data_module +
                     'DataModule')(feat_path=DATA_PATHS[args.data]['feat'],
                                                adj_path=DATA_PATHS[args.data]['adj'],
                                                **vars(args))
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=get_callbacks())
    trainer.fit(task, dm)

def main_densityForecast(args):
    dm = getattr(utils.data, temp_args.data_module +
                     'DataModule')(feat_path=DATA_PATHS[args.data]['feat'],
                                                adj_path=DATA_PATHS[args.data]['adj'],
                                                **vars(args))
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=get_callbacks())
    trainer.fit(task, dm)
    
def main(args):
    rank_zero_info(vars(args))
    globals()['main_' + args.settings](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--data', type=str, help='The name of the dataset',
                        choices=('chengdu'), default='chengdu')
    parser.add_argument('--model-name', type=str, help='The name of the model for spatiotemporal prediction',
                        default='GCN')
    parser.add_argument('--settings', type=str, help='The type of tasks, e.g. forecast task',
                        choices=('forecast', 'densityForecast'), default='forecast')
    parser.add_argument('--data-module', type=str, help='Determine if data has sequential feature',
                        choices=('NS', 'S'), default='NS')

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.data_module +
                     'DataModule').add_data_specific_arguments(parser)
    parser = getattr(
        models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings +
                     'Task').add_task_specific_arguments(parser)
    args = parser.parse_args()
    utils.misc.format_logger(pl._logger)
    main(args)
