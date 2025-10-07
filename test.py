import os
import torch
import argparse

from utils.configs import Config, str2bool
from torch.utils.data import DataLoader
######################################## Pytorch lightning ########################################################
from pytorch_lightning import Trainer, seed_everything
seed_everything(1112)
import tqdm
from pytorch_lightning.plugins import DDPPlugin
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
    parser.add_argument('--weights', default='Summaries/summe_head2_layer3/summe/summe_split0/best_rho_model/epoch=122-val_sRho=0.214.ckpt', type=str, help='Path to weights')
    parser.add_argument('--result_dir', default='Summaries/summe_head2_layer3/summe/', type=str)
    parser.add_argument('--pt_path', type=str, default='llama_emb/summe_sum/')
    
    opt = parser.parse_args()
    kwargs = vars(opt)
    config = Config(**kwargs)

    if 'summe' in config.dataset:
        from utils.summe_dataset import SumMeLLaMADataset, ValBatchCollator
        test_dataset = SumMeLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding = config.pt_path)
    elif 'tvsum' in config.dataset:
        from utils.tvsum_dataset import TVSumLLaMADataset, ValBatchCollator
        test_dataset = TVSumLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding = config.pt_path)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32, collate_fn = ValBatchCollator(), pin_memory=True)

    model = LLMVS.load_from_checkpoint(config.weights, config = config)

    model.cuda()
    model.eval()

    trainer = Trainer(
                    gpus=-1,
                    accelerator='ddp',
                    max_epochs=opt.epochs,
                    accumulate_grad_batches=2,
                    precision=16,
                    gradient_clip_val=0.01,
                    benchmark=True,
                    deterministic=False,
                    progress_bar_refresh_rate=100,
                    log_every_n_steps=1,
                    plugins=DDPPlugin(find_unused_parameters=False),
                    )

    results = trainer.test(model,test_loader, ckpt_path=config.weights)
