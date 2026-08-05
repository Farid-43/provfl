#!/bin/bash
# =====================================================================
# New Defense Experiments for ProVFL
# 5 new defenses: grad_clip, gauss_noise, dp_gauss, grad_sparse, random_proj
# Target: Adult dataset, property=sex
# =====================================================================

run_defense(){
    python vfl_pia_defense.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --defense $4 \
        --d_para $5 \
        --d_para2 ${6:-1.0}
}

db=adult
p=sex

# ======================== 1. Gradient Clipping ========================
echo "===== Gradient Clipping ====="
run_defense $db $p 0 grad_clip 0.1 &
run_defense $db $p 0 grad_clip 0.5 &
run_defense $db $p 0 grad_clip 1.0 &
run_defense $db $p 0 grad_clip 2.0 &
wait

# ======================== 2. Gaussian Noise ========================
echo "===== Gaussian Noise ====="
run_defense $db $p 0 gauss_noise 0.001 &
run_defense $db $p 0 gauss_noise 0.01 &
run_defense $db $p 0 gauss_noise 0.05 &
run_defense $db $p 0 gauss_noise 0.1 &
wait

# ======================== 3. DP Gaussian (Clip + Noise) ========================
echo "===== DP Gaussian (Clip + Noise) ====="
run_defense $db $p 0 dp_gauss 1.0 0.1 &
run_defense $db $p 0 dp_gauss 1.0 0.5 &
run_defense $db $p 0 dp_gauss 0.5 0.5 &
run_defense $db $p 0 dp_gauss 0.5 1.0 &
wait

# ======================== 4. Gradient Sparsification ========================
echo "===== Gradient Sparsification ====="
run_defense $db $p 0 grad_sparse 0.125 &
run_defense $db $p 0 grad_sparse 0.25 &
run_defense $db $p 0 grad_sparse 0.5 &
run_defense $db $p 0 grad_sparse 0.75 &
wait

# ======================== 5. Random Projection ========================
echo "===== Random Projection ====="
run_defense $db $p 0 random_proj 12 &
run_defense $db $p 0 random_proj 8 &
run_defense $db $p 0 random_proj 4 &
run_defense $db $p 0 random_proj 2 &
wait

echo "===== All defense experiments completed ====="
