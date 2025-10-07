import os
import torch
import argparse

from utils.configs import Config, str2bool
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
######################################## Pytorch lightning ########################################################
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
seed_everything(1112)
from pytorch_lightning.loggers import TensorBoardLogger
from networks.model import LLMVS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'summe_head2_layer3', help = 'the name of the model')
    parser.add_argument('--dataset', type = str, default = 'summe', help = 'the name of the dataset')
    parser.add_argument('--split_idx', type = int, default = 0, help = 'the split index')
    parser.add_argument('--epochs', type = int, default = 200, help = 'the number of training epochs')
    parser.add_argument('--reduced_dim', type = int, default = 2048)
    parser.add_argument('--num_heads', type = int, default = 2)
    parser.add_argument('--num_layers', type = int, default = 3)
    parser.add_argument('--tag', type = str, default = 'summe_split0')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'the learning rate')
    parser.add_argument('--pt_path', type=str, default='llama_emb/summe_sum/')

    
    opt = parser.parse_args()
    kwargs = vars(opt)
    config = Config(**kwargs)

    if config.dataset == 'summe':
        from utils.summe_dataset import SumMeLLaMADataset, TrainBatchCollator, ValBatchCollator
        train_dataset = SumMeLLaMADataset(mode='train', split_idx=config.split_idx, llama_embedding = config.pt_path)
        val_dataset = SumMeLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding = config.pt_path)
    elif config.dataset == 'tvsum':
        from utils.tvsum_dataset import TVSumLLaMADataset, TrainBatchCollator,ValBatchCollator
        train_dataset = TVSumLLaMADataset(mode='train', split_idx=config.split_idx, llama_embedding = config.pt_path)
        val_dataset = TVSumLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding = config.pt_path)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn = TrainBatchCollator(), pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn = ValBatchCollator(), pin_memory=True, persistent_workers=True)

    
    model = LLMVS(config = config)
    model.cuda()
        
    best_rho_model = '{}/best_rho_model'.format(config.save_dir_root)
    best_tau_model = '{}/best_tau_model'.format(config.save_dir_root)

    checkpoint_callback_rho = ModelCheckpoint(
    monitor='val_sRho',
    dirpath= best_rho_model,
    filename='{epoch:02d}-{val_sRho:.3f}',
    save_top_k=1,
    save_last=True,
    mode='max',
    )

    checkpoint_callback_tau = ModelCheckpoint(
    monitor='val_kTau',
    dirpath= best_tau_model,
    filename='{epoch:02d}-{val_kTau:.3f}',
    save_top_k=1,
    save_last=True,
    mode='max',
    )

    trainer = Trainer(
                    gpus=-1,
                    accelerator='ddp',
                    max_epochs=opt.epochs,
                    accumulate_grad_batches=2,
                    precision=16,
                    gradient_clip_val=0.01,
                    callbacks=[checkpoint_callback_rho, checkpoint_callback_tau],
                    benchmark=True,
                    deterministic=False,
                    val_check_interval=0.5,
                    progress_bar_refresh_rate=100,
                    profiler="simple",
                    log_every_n_steps=4,
                    plugins=DDPPlugin(find_unused_parameters=False),
                    )

    trainer.validate(model,val_loader)
    trainer.fit(model, train_loader, val_loader)