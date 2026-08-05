# ProVFL
This is the repository for the paper _ProVFL: Property Inference Attacks against Vertical Federated Learning_.

### Getting started
Let's start by installing all the dependencies.
```shell
pip3 install -r requirement.txt
```

### Files and Arguments

We use `*_baseline.py`, `*_passive.py` and `*_active.py` to refer to the baseline, ProVFL-B and ProVFL-M attacks against VFL.
The key arguments and their usage are listed below.
- `--dataset` (`adult|census|bankmk|health|lawschool`): Defined in `dataloader.py`.
- `--property`: Depend on specific datasets as shown in `dataloader.py`.
- `--sampling_size`: The half size of the auxiliary dataset, i.e., $|D_{aux}|/2$.
- `--attack_epoch`: The epoch when PIAs are conducted, i.e., $e_{SEC}$.
- `--interpolate`: The size of the attack training set, i.e., ${R}$.
- `--target_num`: The size of the distribution query, i.e., $Q$.
- `--interval`: The fraction interval, i.e., $q$.
- `--aligned`: Use an aligned or misaligned auxiliary dataset.
- `--use_LR`: Use LR strategy if set 1.
- `--use_MR`: Use MR strategy if set 1.
- `--act_weight`: The hyperparameter $\alpha$ in MR strategy.   

### Run

Take the _Adult-sex_ property as an example.

- Baseline:
```shell
    python vfl_pia_baseline.py --dataset adult --property sex --gpu 0 --attack_epoch 18
```

- ProVFL-B:
```shell
    python vfl_pia_passive.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100
```

- ProVFL-M:  

when adopting MR strategy:
```shell
    python vfl_pia_active.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --use_MR 1 --act_weight 1.0
```
when adopting LR and MR strategies:
```shell
    python vfl_pia_active.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --use_LR 1 --use_MR 1 --act_weight 1.0
```

- Defense
```shell
    python vfl_pia_defense.py --dataset adult --property sex --gpu 0 --attack_epoch 18 --sampling_size 2000 --interpolate 200 --target_num 100 --defense shuffle --d_para 0.1
```

In addition, we provide scripts for readers: `script_attack.sh` includes three attacks on various datasets and properties, `script_defense.sh` involves four defenses against VFL-B on the Adult dataset, and `script_ablation.sh` refers to commands used to explore impacts of different settings.

### Dataset Preparation

1. [Adult](https://archive.ics.uci.edu/dataset/2/adult)
2. [Census](file:///D:/Downloads/census+income+kdd/census-income.html)
3. [Bank Markerting](https://archive.ics.uci.edu/dataset/222/bank+marketing)


### Special Credits
Some of the code in this repository is based on the following amazing works.:

[1] [Label Inference Attacks Against Vertical Federated Learning](https://github.com/FuChong-cyber/label-inference-attacks)

[2] [SNAP: Efficient Extraction of Private Properties with Poisoning](https://github.com/johnmath/snap-sp23.git)

[3] [Uncertainty loss](https://github.com/yaringal/multi-task-learning-example.git)
