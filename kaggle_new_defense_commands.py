# =====================================================================
# Kaggle Notebook Cells for New Defense Experiments
# Copy-paste these commands into your Kaggle notebook cells
# =====================================================================

# --- Cell 1: Run Gradient Clipping defense (4 param values) ---
# d_para = max_norm (L2 clip bound). Smaller = stronger defense.
# Recommended start: 0.5 (moderate defense, minimal accuracy loss)
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_clip --d_para 0.1
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_clip --d_para 0.5
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_clip --d_para 1.0
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_clip --d_para 2.0

# --- Cell 2: Run Gaussian Noise defense (4 param values) ---
# d_para = sigma (noise std dev). Larger = stronger defense.
# Recommended start: 0.01 (moderate defense)
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense gauss_noise --d_para 0.001
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense gauss_noise --d_para 0.01
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense gauss_noise --d_para 0.05
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense gauss_noise --d_para 0.1

# --- Cell 3: Run DP Gaussian (Clip + Noise) defense ---
# d_para = clip_norm (C), d_para2 = noise_multiplier (sigma). noise_std = sigma * C
# This is the strongest defense — formal DP guarantees
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense dp_gauss --d_para 1.0 --d_para2 0.1
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense dp_gauss --d_para 1.0 --d_para2 0.5
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense dp_gauss --d_para 0.5 --d_para2 0.5
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense dp_gauss --d_para 0.5 --d_para2 1.0

# --- Cell 4: Run Gradient Sparsification defense (4 param values) ---
# d_para = keep_ratio (fraction of gradient dims to keep). Smaller = stronger defense.
# With hidden_dim=16: 0.25 keeps 4 dims, 0.125 keeps 2 dims
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_sparse --d_para 0.125
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_sparse --d_para 0.25
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_sparse --d_para 0.5
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense grad_sparse --d_para 0.75

# --- Cell 5: Run Random Projection defense (4 param values) ---
# d_para = proj_dim (dimension of the random subspace). Smaller = stronger defense.
# With hidden_dim=16: try 12, 8, 4, 2
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense random_proj --d_para 12
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense random_proj --d_para 8
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense random_proj --d_para 4
!python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense random_proj --d_para 2