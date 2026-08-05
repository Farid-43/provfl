# baseline / ProVFL-B / ProVFL-M

# ============================== baseline =======================
baseline_attack(){
    python vfl_pia_baseline.py \
        --dataset $1 \
        --property $2 \
        --gpu $3

}

baseline_attack adult sex 0 &
baseline_attack adult race 1 &
baseline_attack adult workclass 2 &
wait 

baseline_attack bankmk month 3 &
baseline_attack bankmk marital 1 & 
baseline_attack bankmk contact 2 &
wait

baseline_attack health sex 0 &
baseline_attack health age 3 &
wait

baseline_attack lawschool race 0 &
baseline_attack lawschool resident 1 &
baseline_attack lawschool gender 2 &
wait

baseline_attack census sex 0 &
baseline_attack census race 1 &
baseline_attack census education 2 &
wait


# ================================== ProVFL-B ============================
passive_attack(){
    python vfl_pia_passive.py \
        --dataset $1 \
        --property $2 \
        --gpu $3
}

passive_attack adult sex 0 &
passive_attack adult race 1 &
passive_attack adult workclass 2 &

passive_attack bankmk month 3 &
passive_attack bankmk marital 1 & 
passive_attack bankmk contact 2 &
wait

passive_attack health sex 0 &
passive_attack health age 3 &

passive_attack lawschool race 0 & 
passive_attack lawschool resident 1 &
passive_attack lawschool gender 2 &
wait

passive_attack census sex 0 &
passive_attack census race 1 &
passive_attack census education 2 &
wait

# ================================== ProVFL-M ============================

# only use LR
active_attack_LR(){
    python vfl_pia_active.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --use_LR 1
}

# only use MR
active_attack_MR(){
    python vfl_pia_active.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --use_MR 1 --act_weight $4
}

# use both MR and LR
active_attack_Double(){
    python vfl_pia_active.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --use_LR 1 \
        --use_MR 1 --act_weight $4
}


active_attack_LR adult sex 0 &
active_attack_LR adult race 1 &
active_attack_LR adult workclass 2 &
wait 

active_attack_LR bankmk month 3 &
active_attack_LR bankmk marital 1 & 
active_attack_LR bankmk contact 2 &
wait

active_attack_LR health sex 0 &
active_attack_LR health age 3 &
wait

active_attack_LR lawschool race 0 &
active_attack_LR lawschool resident 1 &
active_attack_LR lawschool gender 2 &
wait

active_attack_LR census sex 0 &
active_attack_LR census race 1 &
active_attack_LR census education 2 &
wait 

arr=(0.5 1.0 2.0)
for weight in "${arr[@]}"
do
    # ---------------------- MR -----------------------------
    active_attack_MR adult sex 0 $weight &
    active_attack_MR adult race 1 $weight &
    active_attack_MR adult workclass 2 $weight &
    wait

    active_attack_MR bankmk month 3 $weight &
    active_attack_MR bankmk marital 1 $weight & 
    active_attack_MR bankmk contact 2 $weight &
    wait

    active_attack_MR health sex 0 $weight &
    active_attack_MR health age 3 $weight &
    wait

    active_attack_MR lawschool race 0 $weight &
    active_attack_MR lawschool resident 1 $weight &
    active_attack_MR lawschool gender 2 $weight &
    wait

    active_attack_MR census sex 0 $weight &
    active_attack_MR census race 1 $weight &
    active_attack_MR census education 2 $weight &
    wait 

    # ---------------------- MR + LR -----------------------------
    active_attack_Double adult sex 0 $weight &
    active_attack_Double adult race 1 $weight &
    active_attack_Double adult workclass 2 $weight &
    wait

    active_attack_Double bankmk month 3 $weight &
    active_attack_Double bankmk marital 1 $weight & 
    active_attack_Double bankmk contact 2 $weight &
    wait

    active_attack_Double health sex 0 $weight &
    active_attack_Double health age 3 $weight &
    wait

    active_attack_Double lawschool race 0 $weight &
    active_attack_Double lawschool resident 1 $weight &
    active_attack_Double lawschool gender 2 $weight &
    wait

    active_attack_Double census sex 0 $weight &
    active_attack_Double census race 1 $weight &
    active_attack_Double census education 2 $weight &
    wait

done

