#2024-4-11 new defense

new_defense(){
    python vfl_pia_defense.py \
        --dataset $1 \
        --property $2 \
        --gpu $3 \
        --defense $4 \
        --d_para $5
}



db=adult
p=sex
new_defense $db $p 1 withdraw 0.1 &
new_defense $db $p 2 withdraw 0.2 &
new_defense $db $p 3 withdraw 0.5 &

p=workclass
new_defense $db $p 1 withdraw 0.1 &
new_defense $db $p 2 withdraw 0.2 &
new_defense $db $p 3 withdraw 0.5 &

p=race
new_defense $db $p 1 withdraw 0.1 &
new_defense $db $p 2 withdraw 0.2 &
new_defense $db $p 3 withdraw 0.5 & 
wait
  

db=adult
p=sex
new_defense $db $p 0 no 0.0 &
new_defense $db $p 1 lap_noise 0.1 &
new_defense $db $p 0 lap_noise 0.01 &
new_defense $db $p 2 lap_noise 0.001 &
new_defense $db $p 3 lap_noise 0.0001 &


new_defense $db $p 1 ppdl 0.10 &
new_defense $db $p 0 ppdl 0.25 &
new_defense $db $p 2 ppdl 0.50 &
new_defense $db $p 3 ppdl 0.75 &

wait

p=workclass
new_defense $db $p 0 no 0.0 &
new_defense $db $p 0 lap_noise 0.1 &
new_defense $db $p 1 lap_noise 0.01 &
new_defense $db $p 2 lap_noise 0.001 &
new_defense $db $p 3 lap_noise 0.0001 &


new_defense $db $p 1 ppdl 0.10 &
new_defense $db $p 0 ppdl 0.25 &
new_defense $db $p 2 ppdl 0.50 &
new_defense $db $p 3 ppdl 0.75 &

wait

p=race
new_defense $db $p 0 no 0.0 &
new_defense $db $p 1 lap_noise 0.1 &
new_defense $db $p 2 lap_noise 0.01 &
new_defense $db $p 3 lap_noise 0.001 &
new_defense $db $p 0 lap_noise 0.0001 &

 

new_defense $db $p 1 ppdl 0.10 &
new_defense $db $p 0 ppdl 0.25 &
new_defense $db $p 2 ppdl 0.50 &
new_defense $db $p 3 ppdl 0.75 &

wait


db=adult
p=sex
new_defense $db $p 1 shuffle 0.1 &
new_defense $db $p 2 shuffle 0.2 &
new_defense $db $p 3 shuffle 0.5 &

p=workclass
new_defense $db $p 1 shuffle 0.1 &
new_defense $db $p 2 shuffle 0.2 &
new_defense $db $p 3 shuffle 0.5 &

p=race
new_defense $db $p 1 shuffle 0.1 &
new_defense $db $p 2 shuffle 0.2 &
new_defense $db $p 3 shuffle 0.5 &
wait