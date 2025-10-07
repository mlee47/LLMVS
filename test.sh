CUDA_VISIBLE_DEVICES=0,1,2,3,4 python test_splits.py --dataset summe --weights rho \
        --result_dir Summaries/summe_head2_layer3/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python test_splits.py --dataset tvsum --weights rho \
        --result_dir Summaries/tvsum_head2_layer3/ \
        --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048