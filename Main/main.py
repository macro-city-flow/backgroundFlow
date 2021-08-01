import argparse
from utils.callbacks import SendMessageCallback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.misc

def main(args):

    rank_zero_info(vars(args))
    dm = getattr(utils.data, temp_args.data_module +
                 'DataModule')(**vars(args))

    #model = get_model(args, dm)
    model = getattr(models, args.model_name)(**vars(args))

    task = getattr(tasks, args.settings + 'Task')(model=model, feat_max_val=dm.feat_max_val,
                                                  **vars(args))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='train_loss')
    print_best_epoch_metrics_callback = utils.callbacks.PrintBestEpochMetricsCallback(
        monitor='train_loss')
    send_message_callback = SendMessageCallback(monitor='train_loss')
    callbacks = [checkpoint_callback,
                 print_best_epoch_metrics_callback, send_message_callback]

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks)

    trainer.fit(task, dm)


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
