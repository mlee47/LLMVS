CUDA_VISIBLE_DEVICES=2,3,5,6,1 python train.py --tag summe_split0 --model summe_head2_layer3 --lr 0.000119 --epochs 200 --dataset summe --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 0 --pt_path 'llama_emb/summe_sum/'
CUDA_VISIBLE_DEVICES=2,3,5,6,1 python train.py --tag summe_split1 --model summe_head2_layer3 --lr 0.000119 --epochs 200 --dataset summe --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 1 --pt_path 'llama_emb/summe_sum/'
CUDA_VISIBLE_DEVICES=2,3,5,6,1 python train.py --tag summe_split2 --model summe_head2_layer3 --lr 0.000119 --epochs 200 --dataset summe --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 2 --pt_path 'llama_emb/summe_sum/'
CUDA_VISIBLE_DEVICES=2,3,5,6,1 python train.py --tag summe_split3 --model summe_head2_layer3 --lr 0.000119 --epochs 200 --dataset summe --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 3 --pt_path 'llama_emb/summe_sum/'
CUDA_VISIBLE_DEVICES=2,3,5,6,1 python train.py --tag summe_split4 --model summe_head2_layer3 --lr 0.000119 --epochs 200 --dataset summe --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 4 --pt_path 'llama_emb/summe_sum/'


# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train.py --tag tvsum_split0 --model tvsum_head2_layer3 --lr 0.00007 --epochs 200 --dataset tvsum --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 0 --pt_path 'llama_emb/tvsum_sum/'
# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train.py --tag tvsum_split1 --model tvsum_head2_layer3 --lr 0.00007 --epochs 200 --dataset tvsum --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 1 --pt_path 'llama_emb/tvsum_sum/'
# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train.py --tag tvsum_split2 --model tvsum_head2_layer3 --lr 0.00007 --epochs 200 --dataset tvsum --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 2 --pt_path 'llama_emb/tvsum_sum/'
# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train.py --tag tvsum_split3 --model tvsum_head2_layer3 --lr 0.00007 --epochs 200 --dataset tvsum --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 3 --pt_path 'llama_emb/tvsum_sum/'
# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train.py --tag tvsum_split4 --model tvsum_head2_layer3 --lr 0.00007 --epochs 200 --dataset tvsum --reduced_dim 2048 --num_heads 2 --num_layers 3 --split_idx 4 --pt_path 'llama_emb/tvsum_sum/'
