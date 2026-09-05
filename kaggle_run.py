#!/usr/bin/env python
"""
Resume-safe campaign runner for the ProVFL defense experiments on Kaggle.

Kaggle gives a hard 12-hour session ceiling, so a campaign is split into named
sessions of roughly 20 invocations each. Every invocation that finishes writes a
marker into runs_done/; re-running the same session skips completed jobs, so a
crash at hour 11 costs one job, not the session.

Usage on Kaggle
---------------
  !python kaggle_run.py --list                       # see the plan, no GPU used
  !python kaggle_run.py --session s1 --smoke         # 1 seed / 3 epochs sanity pass
  !python kaggle_run.py --session s1 --part 1/2      # first half, ~4h
  !python kaggle_run.py --session s1 --part 2/2      # second half, next notebook run
  !python kaggle_run.py --session s1b                # rest of the comparison table
  !python kaggle_run.py --session s1c                # confound probes + ceiling proof
  !python kaggle_run.py --session s1g --dataset census   # same decomposition, 2nd dataset

Which session, and why (state as of 2026-09-06)
----------------------------------------------
s1 and s1b are done for adult/sex. Recombining their per-channel rows with
analyze_clean_channel.py showed the C1 adversary keeps its full accuracy from
a_output and a_grad alone -- tensors it never transmits -- so wire perturbation
cannot bound this attack. That result sets the order of everything left:

  s1c  first. Two ~5-min probes decide whether the ceiling is a fact about VFL
       or an artefact of adult's relationship_*/marital_* columns, and norm_align
       supplies the constructive half (provably zero norm leakage on b_output).
  s1g  second, once per extra (dataset, property). A single-dataset ceiling will
       not survive review; three consistent replications will.
  s2/s3 are deprecated -- they sweep sigma inside the range s1 measured as inert.
       Selecting one prints why. s4 (the utility/epsilon table) still stands.

Splitting a session
-------------------
s1 is the long one (20 jobs, ~8.5h estimated) and Kaggle sometimes kills a session
before the 12h mark. --part N/M takes a contiguous slice so each notebook run is
short: --part 1/2 then --part 2/2 covers the same 20 jobs in two ~4h runs. Both
parts append to the same res_s1_<dataset>.csv and share runs_done/, so the analyzer
sees one dataset either way and nothing is repeated if a part is re-run.

Every job is launched with --gpu (default 0). The training scripts choose their
device with torch.cuda.is_available(), so if Kaggle's accelerator is off they run on
CPU without complaining -- the runner therefore prints the device before the first
job and warns loudly when CUDA is missing.

Each session writes its own results file (res_<session>_<dataset>.csv) because
utils.write_to_csv appends without a header once a file exists -- mixing schemas
into one file silently misaligns columns.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

DEFENSE = 'vfl_pia_defense.py'      # passive attacker, XGB only, 5 rows per run
ACTIVE = 'vfl_pia_active.py'        # MR/LR attacker, DT+XGB+ensemble, 10 rows per run

# Confound probes. Neither trains a VFL model, so both are minutes rather than
# hours, and neither takes --n_seeds/--out_csv: they write their own fixed files
# (ab_feat_correlation.csv, raw_res_baseline_<dataset>.csv). They live under
# ablation/, which has no __init__.py, so they are launched as namespace-package
# modules -- `python ablation/x.py` would put ablation/ on sys.path[0] and
# `from dataloader import get_data` would fail.
CORRELATION = 'ablation/test_ab_correlation.py'   # Pearson r(attacker feature, property)
RAW_BASELINE = 'ablation/test_ab_raw_baseline.py'  # the attack on RAW a-side features
PROBE_SCRIPTS = (CORRELATION, RAW_BASELINE)

MARKER_DIR = 'runs_done'
LOG_CSV = 'kaggle_run_log.csv'

# Per-job wall clock, recalibrated against the 20-job s1 campaign of 2026-09-05
# (res/1st part, res/2nd part). Measured minutes: DEFENSE 11.6-12.4 except
# lap_noise at 44.9; ACTIVE 11.3-13.5. The previous 22/34 estimates were ~2x
# pessimistic, which made the whole campaign look like 23.7 h when it is ~10 h
# and made the budget guard stop sessions that had hours left.
EST_MIN = {DEFENSE: 14, ACTIVE: 16, CORRELATION: 3, RAW_BASELINE: 8}

# Mechanisms whose cost is not the training loop. DPLaplacianNoiseApplyer draws
# noise per coordinate in python, so lap_noise ran 3.7x the other DEFENSE jobs;
# dp_gc_ppdl is worse still, which is why ppdl stays out of every session.
EST_MIN_OVERRIDE = {'lap_noise': 50, 'ppdl': 240}

# EST_MIN was measured on adult only, and the other datasets are not the same size.
# From config.json and dataloader.py's docstrings:
#
#   adult    44 355 rows x 111 dims, 20 epochs   <- the calibration point, factor 1.0
#   bankmk   45 211 rows x  51 dims, 30 epochs   <- same rows, fewer dims, 1.5x epochs
#   census  299 285 rows x 511 dims, 30 epochs   <- 6.7x rows, 4.6x dims, 1.5x epochs
#
# These factors are deliberately rough: they are there so `--list` and the budget guard
# stop implying a census session costs what an adult session costs. Replace them with
# measured numbers after the first census job the way EST_MIN itself was replaced --
# a wrong-by-2x estimate that is *labelled* as an estimate costs nothing, but a census
# session planned at adult prices is how a 12 h Kaggle window gets truncated.
DATASET_FACTOR = {'adult': 1.0, 'bankmk': 1.5, 'census': 8.0,
                  'lawschool': 1.5, 'health': 2.0}


def est_of(j):
    """Estimated minutes for one job, honouring per-defense and per-dataset scaling."""
    base = EST_MIN_OVERRIDE.get(j['flags'].get('defense'), EST_MIN[j['script']])
    return base * DATASET_FACTOR.get(j['flags'].get('dataset', 'adult'), 1.0)

def job(jid, script, **flags):
    """One invocation. flags become --key value pairs; None values are dropped."""
    return {'id': jid, 'script': script,
            'flags': {k: v for k, v in flags.items() if v is not None}}


def session_s1(a):
    """
    Session 1 -- corrected baselines, side/scope decomposition, the D4/D5 ports,
    and the extended Gaussian sweep.

    Three questions this answers, none of which the existing 2400-row dataset can:
      1. how much of the old numbers came from perturbing a_grad/a_output, which no
         defender can reach under C1 (defend_scope both vs victim_only)
      2. whether the protection lives on the forward or the backward channel
         (defend_side output vs grad vs both) -- only 'output' is unilaterally
         deployable by the victim
      3. whether gauss_noise above d_para=0.05 reaches dp_gauss's privacy level
         while keeping AUC, which decides whether the paper can claim Pareto
         dominance outright or only in the MAE<0.1 regime
    """
    d, p, out = a.dataset, a.property, 'res_s1_%s.csv' % a.dataset
    jobs = [
        # (1) fresh undefended baseline + the norm probe that sets --norm_threshold
        job('s1_none_probe', DEFENSE, defense='None', log_norms=1, out_csv=out),
        # (2) side decomposition at the reference strength
        job('s1_gauss_out_005', DEFENSE, defense='gauss_noise', defend_side='output',
            out_para=0.05, defend_scope='victim_only', out_csv=out),
        job('s1_gauss_grad_005', DEFENSE, defense='gauss_noise', defend_side='grad',
            d_para=0.05, defend_scope='victim_only', out_csv=out),
        job('s1_gauss_both_005', DEFENSE, defense='gauss_noise', defend_side='both',
            d_para=0.05, out_para=0.05, defend_scope='victim_only', out_csv=out),
        # (3) the undeployable upper bound, for the inflation delta
        job('s1_gauss_both_005_scopeboth', DEFENSE, defense='gauss_noise',
            defend_side='both', d_para=0.05, out_para=0.05, defend_scope='both',
            out_csv=out),
        # (4) does gauss_noise reach dp_gauss's privacy without dp_gauss's AUC cost?
        job('s1_gauss_out_010', DEFENSE, defense='gauss_noise', defend_side='output',
            out_para=0.10, defend_scope='victim_only', out_csv=out),
        job('s1_gauss_out_020', DEFENSE, defense='gauss_noise', defend_side='output',
            out_para=0.20, defend_scope='victim_only', out_csv=out),
        job('s1_gauss_out_050', DEFENSE, defense='gauss_noise', defend_side='output',
            out_para=0.50, defend_scope='victim_only', out_csv=out),
    ]

    jobs += [
        # (5) comparators under the corrected scope
        job('s1_dp_gauss_10', DEFENSE, defense='dp_gauss', d_para=1.0, d_para2=1.0,
            defend_side='both', defend_scope='victim_only', out_csv=out),
        # D1: the paper's own Laplacian noise, gradient-side by definition (it assumes
        # a trusted third party injects it)
        job('s1_lap_noise_01', DEFENSE, defense='lap_noise', d_para=0.1,
            defend_side='grad', defend_scope='victim_only', out_csv=out),
        # D5 withdraw and D4 shuffle: implemented in the dataloader / input batch, so
        # side and scope do not apply. D5 is ProVFL's recommended mitigation and had
        # never been run in this harness.
        job('s1_withdraw_02', DEFENSE, defense='withdraw', d_para=0.2, out_csv=out),
        job('s1_withdraw_05', DEFENSE, defense='withdraw', d_para=0.5, out_csv=out),
        job('s1_shuffle_02', DEFENSE, defense='shuffle', d_para=0.2, out_csv=out),
        job('s1_shuffle_05', DEFENSE, defense='shuffle', d_para=0.5, out_csv=out),
    ]

    # The same questions against the strongest attacker (MR + LR). plain gauss_noise
    # got *more* protective as the attacker got stronger (ratio 1.68 in the old grid);
    # that has to survive the switch to victim_only scope or the claim dies.
    aout = 'res_s1_active_%s.csv' % a.dataset
    strong = dict(use_MR=1, use_LR=1, out_csv=aout)
    jobs += [
        job('s1a_none_probe', ACTIVE, defense='None', log_norms=1, **strong),
        job('s1a_gauss_out_005', ACTIVE, defense='gauss_noise', defend_side='output',
            out_para=0.05, defend_scope='victim_only', **strong),
        job('s1a_gauss_out_005_scopeboth', ACTIVE, defense='gauss_noise',
            defend_side='output', out_para=0.05, defend_scope='both', **strong),
        job('s1a_gauss_out_020', ACTIVE, defense='gauss_noise', defend_side='output',
            out_para=0.20, defend_scope='victim_only', **strong),
        job('s1a_withdraw_05', ACTIVE, defense='withdraw', d_para=0.5, **strong),
        job('s1a_shuffle_05', ACTIVE, defense='shuffle', d_para=0.5, **strong),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', d)
        j['flags'].setdefault('property', p)
    return jobs

def session_s1b(a):
    """
    Session 1b -- completes the comparison table under the corrected threat model.

    The roadmap's Table 1 numbers for grad_clip / grad_sparse / random_proj were all
    measured at defend_scope='both', defend_side='both', i.e. including perturbations
    of the attacker's own tensors. Those rows cannot appear in the paper as deployable
    defenses, so each mechanism needs a victim_only, output-side re-measurement.

    grad_clip output-side is the interesting one: clipping per-sample embedding L2
    norms attacks the sorted-norm statistic the attack actually consumes, rather than
    burying it in noise. It has never been run on the forward channel.

    Strengths are bracketed rather than tuned because test_defense_func.py's
    param-trap checks show each mechanism has a silent no-op regime (clip above the
    data norm, keep_ratio 1.0, proj_dim >= hidden_dim or < 1). The bracket guarantees
    at least one in-range point without knowing the embedding scale in advance.

    ppdl is deliberately absent: it is gradient-side by definition (undeployable under
    C1) and dp_gc_ppdl's per-coordinate python loop costs hours per run in this harness.
    """
    out = 'res_s1b_%s.csv' % a.dataset
    base = dict(defend_side='output', defend_scope='victim_only', out_csv=out)
    jobs = [
        # own baseline, so this file is self-contained for PrivacyGain
        job('s1b_none_probe', DEFENSE, defense='None', log_norms=1, out_csv=out),
        # embedding-norm clipping: bracket 0.5 / 2.0 / 8.0
        job('s1b_clip_out_05', DEFENSE, defense='grad_clip', out_para=0.5, **base),
        job('s1b_clip_out_20', DEFENSE, defense='grad_clip', out_para=2.0, **base),
        job('s1b_clip_out_80', DEFENSE, defense='grad_clip', out_para=8.0, **base),
        # embedding sparsification
        job('s1b_sparse_out_02', DEFENSE, defense='grad_sparse', out_para=0.2, **base),
        job('s1b_sparse_out_05', DEFENSE, defense='grad_sparse', out_para=0.5, **base),
        # random projection; proj_dim must satisfy 1 <= p < hidden_dim (16 on adult)
        job('s1b_proj_out_04', DEFENSE, defense='random_proj', out_para=4.0, **base),
        job('s1b_proj_out_08', DEFENSE, defense='random_proj', out_para=8.0, **base),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', a.dataset)
        j['flags'].setdefault('property', a.property)
    return jobs


def session_s2(a):
    """
    Session 2 -- Gate 1. Is norm-triggered adaptive noise better than static noise
    at the same average strength?

    --norm_threshold must come from Session 1's norms_*.csv (column
    victim_norm_raw_med at the attack epoch). Everything here is output-side and
    victim_only, i.e. the only configuration a victim can actually deploy.
    """
    thr = a.norm_threshold
    if thr <= 0:
        print('[warn] --norm_threshold is 0, so V1 fires sigma_high on every batch '
              'and V2 self-normalises. Read the median victim_norm_raw out of '
              'norms_defense_%s.csv from session s1 first.' % a.dataset)
    out, aout = 'res_s2_%s.csv' % a.dataset, 'res_s2_active_%s.csv' % a.dataset
    base = dict(defense='gauss_noise', defend_side='output',
                defend_scope='victim_only', norm_threshold=thr, log_norms=1)

    jobs = [
        # static references at the two ends of the adaptive range, so the comparison
        # is against matched strengths and not just against the old d_para=0.05 point
        job('s2_static_low', DEFENSE, out_para=0.005, adaptive_noise=0, **base),
        job('s2_static_high', DEFENSE, out_para=0.05, adaptive_noise=0, **base),
        # V1: hard threshold
        job('s2_v1_005_05', DEFENSE, adaptive_noise=1, sigma_low=0.005,
            sigma_high=0.05, **base),
        job('s2_v1_001_05', DEFENSE, adaptive_noise=1, sigma_low=0.001,
            sigma_high=0.05, **base),
        job('s2_v1_005_20', DEFENSE, adaptive_noise=1, sigma_low=0.005,
            sigma_high=0.20, **base),
        # V2: continuous
        job('s2_v2_a005', DEFENSE, adaptive_noise=2, sigma_low=0.005,
            sigma_high=0.05, sigma_alpha=0.05, **base),
        job('s2_v2_a020', DEFENSE, adaptive_noise=2, sigma_low=0.005,
            sigma_high=0.20, sigma_alpha=0.20, **base),
    ]
    strong = dict(use_MR=1, use_LR=1)
    jobs += [
        job('s2a_static_high', ACTIVE, out_para=0.05, adaptive_noise=0, **base, **strong),
        job('s2a_v1_005_05', ACTIVE, adaptive_noise=1, sigma_low=0.005,
            sigma_high=0.05, **base, **strong),
        job('s2a_v1_005_20', ACTIVE, adaptive_noise=1, sigma_low=0.005,
            sigma_high=0.20, **base, **strong),
        job('s2a_v2_a005', ACTIVE, adaptive_noise=2, sigma_low=0.005,
            sigma_high=0.05, sigma_alpha=0.05, **base, **strong),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', a.dataset)
        j['flags'].setdefault('property', a.property)
        j['flags']['out_csv'] = aout if j['script'] == ACTIVE else out
    return jobs

def session_s3(a):
    """
    Session 3 -- Gate 2. Does a curriculum on sigma help on top of the adaptive
    trigger, and in which direction?

    curriculum=1 ramps sigma up over --warmup_epochs (the roadmap's reading of the
    paper's Fig. 5d: leakage only stabilises after ~epoch 10, so early noise is
    wasted utility). curriculum=2 decays it, the OUTPOST-style opposite. Running
    both makes the direction an ablation row instead of an assumption.
    """
    out, aout = 'res_s3_%s.csv' % a.dataset, 'res_s3_active_%s.csv' % a.dataset
    base = dict(defense='gauss_noise', defend_side='output',
                defend_scope='victim_only', norm_threshold=a.norm_threshold,
                adaptive_noise=a.adaptive_noise, sigma_low=a.sigma_low,
                sigma_high=a.sigma_high, log_norms=1)
    jobs = [
        job('s3_cur_off', DEFENSE, curriculum=0, **base),
        job('s3_cur_up_05', DEFENSE, curriculum=1, warmup_epochs=5, **base),
        job('s3_cur_up_10', DEFENSE, curriculum=1, warmup_epochs=10, **base),
        job('s3_cur_up_15', DEFENSE, curriculum=1, warmup_epochs=15, **base),
        job('s3_cur_down_10', DEFENSE, curriculum=2, warmup_epochs=10, **base),
        # threshold sensitivity: is the win a real frontier shift or a lucky trigger?
        job('s3_thr_low', DEFENSE, curriculum=0,
            **dict(base, norm_threshold=a.norm_threshold * 0.75)),
        job('s3_thr_high', DEFENSE, curriculum=0,
            **dict(base, norm_threshold=a.norm_threshold * 1.25)),
    ]
    strong = dict(use_MR=1, use_LR=1)
    jobs += [
        job('s3a_cur_off', ACTIVE, curriculum=0, **base, **strong),
        job('s3a_cur_up_10', ACTIVE, curriculum=1, warmup_epochs=10, **base, **strong),
        job('s3a_cur_down_10', ACTIVE, curriculum=2, warmup_epochs=10, **base, **strong),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', a.dataset)
        j['flags'].setdefault('property', a.property)
        j['flags']['out_csv'] = aout if j['script'] == ACTIVE else out
    return jobs


def session_s4(a):
    """
    Session s4 -- P6, the utility / epsilon table. (Generalisation moved to s1g; this
    docstring used to claim P7 and no longer does.)

    Four defence settings x {passive, active MR+LR}, all with an undefended baseline in
    the same file so the utility column has something to be a drop *from*.

    Pass --norm_threshold. Its default of 0.0 is not a usable value here: V1 gates on
    victim_norm_raw > threshold, so 0.0 fires sigma_high on literally every batch and
    the 'adaptive' row becomes a second copy of the 'static' row. Session s1's
    norms_*.csv puts the passive median at 3.3451 (peak 3.6018) and the active median
    at 2.7128, so ~3.0-3.3 is the defensible range -- and the fact that the two traces
    disagree is itself a finding: a threshold tuned on the passive trace under-fires
    against the stronger attacker.
    """
    if a.adaptive_noise and not a.norm_threshold:
        print('[warn] --norm_threshold is 0, so V1 fires sigma_high on every batch and '
              's4_*_adaptive duplicates s4_*_static. Pass --norm_threshold 3.1 (session '
              's1 passive median 3.3451, active 2.7128) or --adaptive_noise 0.')
    tag = '%s_%s' % (a.dataset, a.property)
    out, aout = 'res_s4_%s.csv' % tag, 'res_s4_active_%s.csv' % tag
    frozen = dict(defense='gauss_noise', defend_side='output',
                  defend_scope='victim_only', norm_threshold=a.norm_threshold,
                  adaptive_noise=a.adaptive_noise, sigma_low=a.sigma_low,
                  sigma_high=a.sigma_high, curriculum=a.curriculum,
                  warmup_epochs=a.warmup_epochs, log_norms=1)
    jobs = [
        job('s4_%s_none_probe' % tag, DEFENSE, defense='None', log_norms=1),
        job('s4_%s_static' % tag, DEFENSE, defense='gauss_noise',
            defend_side='output', defend_scope='victim_only', out_para=a.sigma_high),
        job('s4_%s_adaptive' % tag, DEFENSE, **frozen),
        job('s4_%s_withdraw_05' % tag, DEFENSE, defense='withdraw', d_para=0.5),
        job('s4a_%s_none' % tag, ACTIVE, defense='None', use_MR=1, use_LR=1),
        job('s4a_%s_static' % tag, ACTIVE, defense='gauss_noise', defend_side='output',
            defend_scope='victim_only', out_para=a.sigma_high, use_MR=1, use_LR=1),
        job('s4a_%s_adaptive' % tag, ACTIVE, use_MR=1, use_LR=1, **frozen),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', a.dataset)
        j['flags'].setdefault('property', a.property)
        j['flags']['out_csv'] = aout if j['script'] == ACTIVE else out
    return jobs


def session_s1c(a):
    """
    Session 1c -- close the two holes a reviewer will put a finger through.

    Session s1 plus analyze_clean_channel.py established that the C1 adversary
    reaches its full accuracy (MAE ~0.0136 passive, ~0.0065 active on adult/sex)
    from a_output and a_grad alone -- tensors it computes inside its own bottom
    model and never transmits. Every mechanism in the literature perturbs the
    wire, so none of them can reach that channel. Two objections remain, and
    neither needs new analysis code, only these runs:

    (1) THE CONFOUND. adult's property column sits at feature index 0, and
        utils.split_data gives the victim data[:, 0:50] and the attacker
        data[:, 50:111]. So sex_Male is exclusively on the victim's side -- good.
        But the attacker's half holds relationship_* and marital_status_*, which
        are near-deterministic proxies for sex. If the attacker can read the
        property off its own RAW features, the ceiling is a dataset artefact
        rather than a property of VFL co-training. test_ab_correlation.py already
        measures exactly this (max/mean/median Pearson r between each attacker
        feature and the property label), and test_ab_raw_baseline.py runs the
        attack without any VFL training. Both are cheap and settle it.

    (2) THE CONSTRUCTIVE HALF. The ceiling is currently an upper bound the tried
        mechanisms never approach. norm_align drives the victim's sorted-norm
        vector to a constant, so b_output leakage is provably zero -- if MAE_all
        still lands on MAE_own, the bound is tight, and that is a theorem-shaped
        statement rather than a table of failures. norm_permute is the placebo
        that must score zero; norm_quant traces the curve between them.

    The high-sigma probe finishes the additive-noise story: s1 stopped at
    out_para=0.50 with MAE_all still 0.0111 and AUC 0.8954. Taking sigma to 1/2/4
    shows whether output-side Gaussian noise EVER helps before utility collapses,
    which is the last place a reader can hope the negative result is a tuning
    failure.
    """
    d, p = a.dataset, a.property
    out = 'res_s1c_%s.csv' % d
    vic = dict(defend_side='output', defend_scope='victim_only', out_csv=out)
    jobs = [
        # (1) the confound probes: ~11 min for both, and they decide whether the
        #     ceiling is a statement about VFL or about this dataset's one-hot columns
        job('s1c_feat_corr', CORRELATION),
        # baseline_pipeline hardcodes SVM+DT+NN and ignores --classifier; the row it
        # writes carries gnd_frac next to *_a_input_pred, so MAE comes straight out
        job('s1c_raw_baseline', RAW_BASELINE),
        job('s1c_none_probe', DEFENSE, defense='None', log_norms=1, out_csv=out),
        # (2) constructive ceiling: provably-zero norm leakage
        job('s1c_align_mean', DEFENSE, defense='norm_align', out_para=-1.0,
            log_norms=1, **vic),
        job('s1c_quant_02', DEFENSE, defense='norm_quant', out_para=2.0, **vic),
        job('s1c_quant_08', DEFENSE, defense='norm_quant', out_para=8.0, **vic),
        # the placebo: same norm multiset, so PrivacyGain must come out ~0.0000
        job('s1c_permute', DEFENSE, defense='norm_permute', out_para=-1.0, **vic),
        # the last of the additive-noise sweep
        job('s1c_gauss_out_100', DEFENSE, defense='gauss_noise', out_para=1.0, **vic),
        job('s1c_gauss_out_200', DEFENSE, defense='gauss_noise', out_para=2.0, **vic),
        job('s1c_gauss_out_400', DEFENSE, defense='gauss_noise', out_para=4.0, **vic),
    ]
    aout = 'res_s1c_active_%s.csv' % d
    strong = dict(use_MR=1, use_LR=1, out_csv=aout)
    jobs += [
        job('s1ca_none', ACTIVE, defense='None', log_norms=1, **strong),
        job('s1ca_align_mean', ACTIVE, defense='norm_align', out_para=-1.0,
            defend_side='output', defend_scope='victim_only', log_norms=1, **strong),
        job('s1ca_gauss_out_200', ACTIVE, defense='gauss_noise', out_para=2.0,
            defend_side='output', defend_scope='victim_only', **strong),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', d)
        j['flags'].setdefault('property', p)
    return jobs


def session_s1g(a):
    """
    Session 1g -- the same decomposition on another (dataset, property).

    The roadmap put generalisation at P6, after the two design gates. The gates
    tested whether norm-triggered adaptive Gaussian noise beats static Gaussian
    noise, and s1 showed the entire family is inert on the only channel a victim
    can reach, so that ordering is now backwards: the result that needs
    replicating is the DECOMPOSITION, and it needs replicating before anything
    else, because a single-dataset finding will not survive review.

    Seven DEFENSE + two ACTIVE jobs, chosen so each of the paper's claims gets one
    supporting row per dataset and nothing else:

      none            the undefended reference (and the norm trace)
      gauss out 0.05  the only unilaterally deployable additive config
      gauss out 0.50  the same at 10x, to pre-empt "you under-tuned it"
      scope=both      the inflation delta -- how much of the old numbers came
                      from perturbing the attacker's own tensors
      withdraw 0.5    ProVFL's own recommended mitigation (D5)
      shuffle 0.5     D4
      norm_align      the provable-zero-leakage point

    Run once per setting: --session s1g --dataset census --property sex, etc. The
    acceptance bar stays the roadmap's: consistent behaviour in >=3 of
    {adult-sex, adult-race, census, bankmk}.
    """
    tag = '%s_%s' % (a.dataset, a.property)
    out, aout = 'res_s1g_%s.csv' % tag, 'res_s1g_active_%s.csv' % tag
    vic = dict(defend_side='output', defend_scope='victim_only')
    jobs = [
        job('s1g_%s_none' % tag, DEFENSE, defense='None', log_norms=1),
        job('s1g_%s_gauss_005' % tag, DEFENSE, defense='gauss_noise',
            out_para=0.05, **vic),
        job('s1g_%s_gauss_050' % tag, DEFENSE, defense='gauss_noise',
            out_para=0.50, **vic),
        job('s1g_%s_scopeboth' % tag, DEFENSE, defense='gauss_noise',
            defend_side='both', d_para=0.05, out_para=0.05, defend_scope='both'),
        job('s1g_%s_withdraw_05' % tag, DEFENSE, defense='withdraw', d_para=0.5),
        job('s1g_%s_shuffle_05' % tag, DEFENSE, defense='shuffle', d_para=0.5),
        job('s1g_%s_align' % tag, DEFENSE, defense='norm_align', out_para=-1.0,
            log_norms=1, **vic),
        job('s1ga_%s_none' % tag, ACTIVE, defense='None', use_MR=1, use_LR=1,
            log_norms=1),
        job('s1ga_%s_align' % tag, ACTIVE, defense='norm_align', out_para=-1.0,
            use_MR=1, use_LR=1, log_norms=1, **vic),
    ]
    for j in jobs:
        j['flags'].setdefault('dataset', a.dataset)
        j['flags'].setdefault('property', a.property)
        j['flags']['out_csv'] = aout if j['script'] == ACTIVE else out
    return jobs


# Sessions whose premise session s1 falsified. They are kept runnable -- the
# adaptive-noise design is still a legitimate ablation and someone will ask for
# the numbers -- but selecting one prints why it is no longer on the critical
# path, so the warning cannot be missed the way a docstring can.
DEPRECATED_SESSIONS = {
    's2': 'Gate 1 asks whether norm-triggered adaptive Gaussian noise beats static '
          'Gaussian noise. Both arms live in sigma 0.005-0.20 on defend_side=output, '
          'and s1 measured PrivacyGain there as -0.0009/-0.0033/+0.0023 at '
          'sigma=0.05/0.10/0.20 (p=0.6150 at the reference point). The comparison is '
          'between two inert conditions. Run s1c and s1g first; if you still want '
          'these numbers, they are an ablation, not a gate.',
    's3': 'Gate 2 tunes a curriculum on top of the Gate 1 trigger. It inherits Gate '
          "1's premise, so the same objection applies with one more free parameter.",
}


SESSIONS = {'s1': session_s1, 's1b': session_s1b, 's1c': session_s1c,
            's1g': session_s1g, 's2': session_s2, 's3': session_s3,
            's4': session_s4}

def marker_path(j, a):
    """Markers are namespaced by dataset/property so s1 on adult and on census
    do not skip each other's jobs."""
    return os.path.join(MARKER_DIR, '%s__%s__%s.done'
                        % (a.dataset, a.property, j['id']))


def build_cmd(j, a):
    probe = j['script'] in PROBE_SCRIPTS
    if '/' in j['script']:
        # namespace-package launch, so the repo root stays on sys.path
        cmd = [sys.executable, '-m', j['script'][:-3].replace('/', '.')]
    else:
        cmd = [sys.executable, j['script']]
    flags = dict(j['flags'])
    flags.setdefault('gpu', a.gpu)
    if a.smoke:
        flags['epochs'] = 3
        flags['attack_epoch'] = 2          # default 18 never fires with 3 epochs
        if not probe:
            flags['n_seeds'] = 1
            flags['out_csv'] = 'res_smoke_%s.csv' % a.dataset
    elif not probe:
        # the probes' parsers have no --n_seeds; they are single-pass by construction
        flags.setdefault('n_seeds', a.n_seeds)
    for k, v in sorted(flags.items()):
        cmd += ['--%s' % k, str(v)]
    return cmd


def gpu_report():
    """
    Loud, early warning if the accelerator is off. Both training scripts pick their
    device with `torch.cuda.is_available()`, so a session started without a GPU does
    not fail -- it silently trains on CPU, where EST_MIN is wildly optimistic and the
    12h ceiling arrives long before the session's jobs do.
    """
    try:
        import torch
    except ImportError:
        print('[warn] torch is not importable here, cannot verify the accelerator')
        return
    if torch.cuda.is_available():
        print('device: cuda -> %s | torch %s | %d visible'
              % (torch.cuda.get_device_name(0), torch.__version__,
                 torch.cuda.device_count()))
    else:
        print('[WARN] torch.cuda.is_available() is False. The scripts will run on CPU '
              'and the 22-34 min/job estimates below do not apply.\n'
              '       On Kaggle: Notebook settings -> Accelerator -> GPU, then restart '
              'and re-run. Markers in %s mean nothing already finished is lost.'
              % MARKER_DIR)


def slice_part(jobs, spec):
    """
    --part N/M -> contiguous slice N of M, so one notebook run stays well under the
    12h ceiling. The order is stable and the baseline probe stays in part 1, and both
    parts append to the same result CSV, so the analyzer sees one dataset either way.
    """
    try:
        n, m = [int(x) for x in str(spec).split('/')]
    except ValueError:
        sys.exit('bad --part %r, expected N/M such as 1/2' % spec)
    if not 1 <= n <= m:
        sys.exit('bad --part %r, need 1 <= N <= M' % spec)
    size = (len(jobs) + m - 1) // m
    out = jobs[(n - 1) * size:n * size]
    if not out:
        sys.exit('--part %s is empty: only %d jobs in this session' % (spec, len(jobs)))
    return out


def log_row(row):
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_job(j, a, idx, total):
    mark = marker_path(j, a)
    if os.path.exists(mark) and not a.force:
        print('[%2d/%2d] SKIP %s (marker exists)' % (idx, total, j['id']))
        return 0, 0.0
    cmd = build_cmd(j, a)
    print('\n[%2d/%2d] RUN  %s\n         %s' % (idx, total, j['id'], ' '.join(cmd)))
    if a.dry_run:
        return 0, 0.0
    t0 = time.time()
    proc = subprocess.run(cmd)
    mins = (time.time() - t0) / 60.0
    log_row({'job': j['id'], 'script': j['script'], 'dataset': a.dataset,
             'property': a.property, 'returncode': proc.returncode,
             'minutes': '%.1f' % mins,
             'finished': time.strftime('%Y-%m-%d %H:%M:%S')})
    if proc.returncode == 0:
        open(mark, 'w').write(' '.join(cmd))
        print('[%2d/%2d] DONE %s in %.1f min' % (idx, total, j['id'], mins))
    else:
        print('[%2d/%2d] FAIL %s rc=%d after %.1f min'
              % (idx, total, j['id'], proc.returncode, mins))
    return proc.returncode, mins

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--session', default='s1', choices=sorted(SESSIONS),
                   help='which campaign session to run')
    p.add_argument('--dataset', default='adult')
    p.add_argument('--property', default='sex')
    p.add_argument('--n_seeds', type=int, default=5,
                   help='seeds per job; must match across scripts for paired tests')
    p.add_argument('--list', action='store_true', help='print the plan and exit')
    p.add_argument('--dry_run', action='store_true', help='print commands, run nothing')
    p.add_argument('--smoke', action='store_true',
                   help='1 seed / 3 epochs into res_smoke_<dataset>.csv, ~2 min per job')
    p.add_argument('--only', nargs='*', default=[],
                   help='run just these job ids (substring match)')
    p.add_argument('--part', default='', metavar='N/M',
                   help='run only slice N of M of the session, e.g. --part 1/2. Use this '
                        'to keep one notebook run short instead of risking a 12h timeout')
    p.add_argument('--gpu', type=int, default=0,
                   help='gpu device id, forwarded to every job (Kaggle: 0)')
    p.add_argument('--force', action='store_true', help='ignore existing markers')
    p.add_argument('--budget_min', type=float, default=690.0,
                   help='stop before exceeding this many minutes (Kaggle ceiling is 720)')
    # frozen-design parameters consumed by sessions s2/s3/s4
    p.add_argument('--norm_threshold', type=float, default=0.0,
                   help='from session s1: median victim_norm_raw in norms_*.csv')
    p.add_argument('--adaptive_noise', type=int, default=1)
    p.add_argument('--sigma_low', type=float, default=0.005)
    p.add_argument('--sigma_high', type=float, default=0.05)
    p.add_argument('--curriculum', type=int, default=0)
    p.add_argument('--warmup_epochs', type=int, default=10)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.session in DEPRECATED_SESSIONS:
        print('[DEPRECATED] session %s -- %s\n' % (a.session, DEPRECATED_SESSIONS[a.session]))
    jobs = SESSIONS[a.session](a)
    if a.only:
        jobs = [j for j in jobs if any(s in j['id'] for s in a.only)]
        if not jobs:
            sys.exit('no job id matches %s' % a.only)
    n_total = len(jobs)
    if a.part:
        jobs = slice_part(jobs, a.part)

    est = sum(est_of(j) for j in jobs)
    if a.smoke:
        est = 2 * len(jobs)
    print('session %s%s | %s/%s | %d jobs%s | est %.0f min (%.1f h)'
          % (a.session, ' part %s' % a.part if a.part else '', a.dataset, a.property,
             len(jobs), ' of %d' % n_total if a.part else '', est, est / 60.0))
    over_budget = est > a.budget_min and not a.smoke
    if over_budget:
        # Printed before the --list return as well: --list is exactly when the session
        # is being planned, and a 17 h census estimate is information you want then,
        # not after the notebook has been saved for background execution.
        print('[warn] estimate %.0f min exceeds the %.0f min budget. Markers make '
              'this resumable -- run the session again next Kaggle session to finish, '
              'or split it with --part 1/2 and --part 2/2.'
              % (est, a.budget_min))
    if a.list:
        for i, j in enumerate(jobs, 1):
            # print the real command, so --list cannot drift from what --session runs
            print('%2d. %-30s %s' % (i, j['id'], ' '.join(build_cmd(j, a)[1:])))
        return 0
    gpu_report()

    if not a.dry_run:
        os.makedirs(MARKER_DIR, exist_ok=True)

    spent, failed, done = 0.0, [], 0
    t_start = time.time()
    for i, j in enumerate(jobs, 1):
        elapsed = (time.time() - t_start) / 60.0
        if elapsed + est_of(j) > a.budget_min and not a.dry_run:
            print('\n[stop] %.0f min elapsed, next job needs ~%d more, budget is %.0f. '
                  'Remaining jobs keep their markers unset -- re-run this exact '
                  'command in the next session to continue.'
                  % (elapsed, est_of(j), a.budget_min))
            break
        rc, mins = run_job(j, a, i, len(jobs))
        spent += mins
        if rc == 0:
            done += 1
        else:
            failed.append(j['id'])

    print('\n=== session %s finished: %d ok, %d failed, %.1f min of work ==='
          % (a.session, done, len(failed), spent))
    if failed:
        print('failed jobs (no marker written, they will re-run): %s' % ', '.join(failed))
    print('results: res_%s*_%s.csv | trace: norms_*_%s.csv | log: %s'
          % (a.session, a.dataset, a.dataset, LOG_CSV))
    print('next: python analyze_defense_results.py "res_%s*.csv" '
          '--filter attack_feat=all --group defense defend_side defend_scope --pareto'
          % a.session)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
