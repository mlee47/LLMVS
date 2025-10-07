import os
import argparse
import glob
import re
from pytorch_lightning import Trainer, seed_everything
seed_everything(1112)
parser = argparse.ArgumentParser("check llama score")
parser.add_argument('--dataset', type=str, default='summe')
parser.add_argument('--weights', default='best_f1score_model', type=str, help='Path to weights')
parser.add_argument('--result_dir', default='Summaries/llama_finetuning/summe/', type=str)
parser.add_argument('--pt_path', type=str, default='llama_emb/summe_sum/0224_partial_7window_note_emb_slicing_fp16_sep/')
parser.add_argument('--reduced_dim', type=int, default=2048)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=3)


args = parser.parse_args()
args.model = args.result_dir.replace('Summaries/', '')
val_kTau = []
val_sRho = []


if 'rho' in args.weights:
    weights = sorted(glob.glob('Summaries/{}/{}/*/best_rho_model/epoch=*'.format(args.model, args.dataset)))
    if os.path.isfile('{}/{}/best_rho_results.txt'.format(args.result_dir, args.dataset)):
        os.remove('{}/{}/best_rho_results.txt'.format(args.result_dir, args.dataset))
    file_name = '{}/{}/best_rho_results.txt'.format(args.result_dir, args.dataset)

if 'tau' in args.weights:
    weights = sorted(glob.glob('Summaries/{}/{}/*/best_tau_model/epoch=*'.format(args.model, args.dataset)))
    if os.path.isfile('{}/{}/best_tau_results.txt'.format(args.result_dir, args.dataset)):
        os.remove('{}/{}/best_tau_results.txt'.format(args.result_dir, args.dataset))
    file_name = '{}/{}/best_tau_results.txt'.format(args.result_dir, args.dataset)

tags =sorted(glob.glob('Summaries/{}/{}/*'.format(args.model, args.dataset)))
tags = [a.split('/')[-1] for a in tags if os.path.isdir(a)]

for i in range(0,5):
    os.system("python test.py --model {} --dataset {} --tag {} --reduced_dim {} --split_idx {} --weights {} --result_dir {} \
        --pt_path {} --num_heads {} --num_layers {} >> {}".format(args.model, args.dataset, tags[i], args.reduced_dim, i, weights[i], args.result_dir, args.pt_path, args.num_heads, args.num_layers, file_name))

file1 = open(file_name, 'r')

Lines = file1.readlines()

for line in Lines:
    if 'val_kTau' in line and 'val_sRho' in line:
        import ast
        result_dict = ast.literal_eval(line.strip())
        val_kTau.append(float(result_dict['val_kTau']))
        val_sRho.append(float(result_dict['val_sRho']))

print('{:.3f}'.format(sum(val_kTau)/len(val_kTau)))
print('{:.3f}'.format(sum(val_sRho)/len(val_sRho)))

with open(file_name, 'a') as f:
    f.write('{:.3f}'.format(sum(val_kTau)/len(val_kTau)))
    f.write('\n')
    f.write('{:.3f}'.format(sum(val_sRho)/len(val_sRho)))