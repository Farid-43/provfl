
# ==================== different regression model ===================
ab_test_clf(){
    python test_ab_clf.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_clf adult sex 0 &
ab_test_clf adult race 0 &
ab_test_clf adult workclass 0 &

ab_test_clf bankmk month 1 &
ab_test_clf bankmk marital 1 &
ab_test_clf bankmk contact 1 &

ab_test_clf health sex 2 &
ab_test_clf health age 2 &

ab_test_clf lawschool race 3 &
ab_test_clf lawschool resident 3 &
ab_test_clf lawschool gender 3 &
wait

# ==================== different target num ===================

ab_test_target_num(){
    python test_ab_query.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_target_num adult sex 0 &
ab_test_target_num adult race 0 &
ab_test_target_num adult workclass 0 &

ab_test_target_num bankmk month 1 &
ab_test_target_num bankmk marital 1 &
ab_test_target_num bankmk contact 1 &

ab_test_target_num health sex 2 &
ab_test_target_num health age 2 &

ab_test_target_num lawschool race 3 &
ab_test_target_num lawschool resident 3 &
ab_test_target_num lawschool gender 3 &
wait

# =================== attack training dataset size ==============

ab_test_attack_dataset(){
    python test_ab_attset.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_attack_dataset adult sex 0 &
ab_test_attack_dataset adult race 0 &
ab_test_attack_dataset adult workclass 0 &

ab_test_attack_dataset bankmk month 1 &
ab_test_attack_dataset bankmk marital 1 &
ab_test_attack_dataset bankmk contact 1 &

ab_test_attack_dataset health sex 2 &
ab_test_attack_dataset health age 2 &

ab_test_attack_dataset lawschool race 3 &
ab_test_attack_dataset lawschool resident 3 &
ab_test_attack_dataset lawschool gender 3 &
wait

# =========================== auxiliary dataset size =================

ab_test_aux_size(){
    python test_ab_aux.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --sampling_size 6000
}

ab_test_aux_size adult sex 0 &
ab_test_aux_size adult race 0 &
ab_test_aux_size adult workclass 0 &

ab_test_aux_size bankmk month 1 &
ab_test_aux_size bankmk marital 1 &
ab_test_aux_size bankmk contact 1 &

ab_test_aux_size health sex 2 &
ab_test_aux_size health age 2 &

ab_test_aux_size lawschool race 3 &
ab_test_aux_size lawschool resident 3 &
ab_test_aux_size lawschool gender 3 &
wait

# ==========================different interval =======================
ab_test_interval(){
    python test_ab_inter.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_interval adult sex 0 &
ab_test_interval adult race 0 &
ab_test_interval adult workclass 0 &

ab_test_interval bankmk month 1 &
ab_test_interval bankmk marital 1 &
ab_test_interval bankmk contact 1 &

ab_test_interval health sex 2 &
ab_test_interval health age 2 &

ab_test_interval lawschool race 3 &
ab_test_interval lawschool resident 3 &
ab_test_interval lawschool gender 3 &
wait

# ========================= different epoch =============================

ab_test_epoch(){
    python test_ab_epoch.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_epoch adult sex 0 &
ab_test_epoch adult race 0 &
ab_test_epoch adult workclass 0 &

ab_test_epoch bankmk month 1 &
ab_test_epoch bankmk marital 1 &
ab_test_epoch bankmk contact 1 &

ab_test_epoch health sex 2 &
ab_test_epoch health age 2 &

ab_test_epoch lawschool race 3 &
ab_test_epoch lawschool resident 3 &
ab_test_epoch lawschool gender 3 &
wait

# ===================== uneven party ===============================

ab_test_uneven_party(){
    python test_ab_uparty.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --k 3 \
        --victim_feature $4
}

ab_test_uneven_party census sex 0 0.1 &
ab_test_uneven_party census race 0 0.1 &
ab_test_uneven_party census education 0 0.1 &

ab_test_uneven_party census sex 1 0.2 &
ab_test_uneven_party census race 1 0.2 &
ab_test_uneven_party census education 1 0.2 &

ab_test_uneven_party census sex 2 0.5 &
ab_test_uneven_party census race 2 0.5 &
ab_test_uneven_party census education 2 0.5 &

ab_test_uneven_party census sex 3 0.8 &
ab_test_uneven_party census race 3 0.8 &
ab_test_uneven_party census education 3 0.8 &
wait

# ============================ k party =============================

ab_test_kparty(){
    python test_ab_kparty.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --k $4
}

kp=4
ab_test_kparty census sex 0 $kp &
ab_test_kparty census race 0 $kp &
ab_test_kparty census education 0 $kp &

kp=6
ab_test_kparty census sex 1 $kp &
ab_test_kparty census race 1 $kp &
ab_test_kparty census education 1 $kp &

kp=8
ab_test_kparty census sex 2 $kp &
ab_test_kparty census race 2 $kp &
ab_test_kparty census education 2 $kp &

kp=10
ab_test_kparty census sex 3 $kp &
ab_test_kparty census race 3 $kp &
ab_test_kparty census education 3 $kp &
wait


# ====================== noise auxiliary dataset =====================

ab_test_noise(){
    python test_ab_noise.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

ab_test_noise adult sex 0 &
ab_test_noise adult race 0 &
ab_test_noise adult workclass 0 &

ab_test_noise bankmk month 1 &
ab_test_noise bankmk marital 1 &
ab_test_noise bankmk contact 1 &

ab_test_noise health sex 2 &
ab_test_noise health age 2 &

ab_test_noise lawschool race 3 &
ab_test_noise lawschool resident 3 &
ab_test_noise lawschool gender 3 &
wait


# ================================ mis baseline ===============================

mis_baseline(){
    python vfl_mis_baseline.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --aligned 0
}

mis_baseline adult sex 0 &
mis_baseline adult race 0 &
mis_baseline adult workclass 0 &

mis_baseline bankmk month 1 &
mis_baseline bankmk marital 1 &
mis_baseline bankmk contact 1 &

mis_baseline health sex 2 &
mis_baseline health age 2 &

mis_baseline lawschool race 3 &
mis_baseline lawschool resident 3 &
mis_baseline lawschool gender 3 &
wait


# ================================== mis passive ==============================

mis_passive(){
    python vfl_mis_passive.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --aligned 0
}

mis_passive adult sex 0 &
mis_passive adult race 0 &
mis_passive adult workclass 0 &

mis_passive bankmk month 1 &
mis_passive bankmk marital 1 &
mis_passive bankmk contact 1 &

mis_passive health sex 2 &
mis_passive health age 2 &

mis_passive lawschool race 3 &
mis_passive lawschool resident 3 &
mis_passive lawschool gender 3 &
wait

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

mis_active_c1 adult sex 0 &
mis_active_c1 adult race 0 &
mis_active_c1 adult workclass 0 &

mis_active_c1 bankmk month 1 &
mis_active_c1 bankmk marital 1 &
mis_active_c1 bankmk contact 1 &

mis_active_c1 health sex 2 &
mis_active_c1 health age 2 &

mis_active_c1 lawschool race 3 &
mis_active_c1 lawschool resident 3 &
mis_active_c1 lawschool gender 3 &
wait

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

mis_active_c2 adult sex 0 &
mis_active_c2 adult race 0 &
mis_active_c2 adult workclass 0 &

mis_active_c2 bankmk month 1 &
mis_active_c2 bankmk marital 1 &
mis_active_c2 bankmk contact 1 &

mis_active_c2 health sex 2 &
mis_active_c2 health age 2 &

mis_active_c2 lawschool race 3 &
mis_active_c2 lawschool resident 3 &
mis_active_c2 lawschool gender 3 &
wait