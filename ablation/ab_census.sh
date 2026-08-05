
# ==================== different regression model ===================
ab_test_clf(){
    python test_ab_clf.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_clf census sex 0 &
ab_test_clf census race 0 &
ab_test_clf census education 0 &


# ==================== different target num ===================

ab_test_target_num(){
    python test_ab_query.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_target_num census sex 1 &
ab_test_target_num census race 1 &
ab_test_target_num census education 1 &

# =================== attack training dataset size ==============

ab_test_attack_dataset(){
    python test_ab_attset.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_attack_dataset census sex 2 &
ab_test_attack_dataset census race 2 &
ab_test_attack_dataset census education 2 &

wait

# =========================== auxiliary dataset size =================

ab_test_aux_size(){
    python test_ab_aux.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --sampling_size 6000
}

ab_test_aux_size census sex 0 &
ab_test_aux_size census race 0 &
ab_test_aux_size census education 0 &

# ==========================different interval =======================
ab_test_interval(){
    python test_ab_inter.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_interval census sex 1 &
ab_test_interval census race 1 &
ab_test_interval census education 1 &

# ========================= different epoch =============================

ab_test_epoch(){
    python test_ab_epoch.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_epoch census sex 2 &
ab_test_epoch census race 2 &
ab_test_epoch census education 2 &
wait

# ====================== noise auxiliary dataset =====================

ab_test_noise(){
    python test_ab_noise.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_noise census sex 0 &
ab_test_noise census race 0 &
ab_test_noise census education 0 &


# ================================ mis baseline ===============================

mis_baseline(){
    python vfl_mis_baseline.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --aligned 0
}

mis_baseline census sex 1 &
mis_baseline census race 1 &
mis_baseline census education 1 &

wait


# ================================== mis passive ==============================

mis_passive(){
    python vfl_mis_passive.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --aligned 0
}

mis_passive census sex 0 &
mis_passive census race 0 &
mis_passive census education 0 &


# ========================== mis active ===================================
mis_active_c1(){
    python vfl_mis_active.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --aligned 0 \
        --use_LR 1 \
        --use_MR 1 \
        --act_weight 1.0
}

mis_active_c1 census sex 1 &
mis_active_c1 census race 1 &
mis_active_c1 census education 1 &


# ======================== mis active ==============================

mis_active_c2(){
    python vfl_mis_active.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --aligned 0 \
        --use_LR 0 \
        --use_MR 1 \
        --act_weight 1.0
}

mis_active_c2 census sex 0 &
mis_active_c2 census race 0 &
mis_active_c2 census education 0 &

wait