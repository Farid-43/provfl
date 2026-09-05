#!/usr/bin/env python
"""
Clean-channel ceiling: what the C1 adversary still infers when the victim's
forward embedding is ignored entirely.

Why this is computable without any GPU time
-------------------------------------------
my_utils/pia_func.property_ensemble does not train a joint model. It runs the
same sampling attack once per channel and averages the four estimates:

    pred_frac('all') = mean(pred b_grad, pred b_output, pred a_grad, pred a_output)

Every one of those four per-channel estimates is already written to the result
CSV as its own row (classifier='XGB', attack_feat=<channel>). So the ensemble
that *drops* b_output is a three-term mean over numbers we already have:

    pred_frac('clean') = mean(pred b_grad, pred a_grad, pred a_output)

That matters for the threat model. Under C1 the adversary is the active party:
it computes a_output and a_grad locally and it computes b_grad itself before
sending it down. The only tensor a victim-side forward defense can touch is
b_output. Therefore for ANY victim-side forward transform T,

    MAE_all(T) <= (something that converges to) MAE_clean   as T -> maximally destructive

because the adversary can always discard the defended channel and fall back to
the three it owns. MAE_clean is the floor the defense cannot push the attacker
above -- an information-theoretic-style ceiling on protection, measured rather
than argued.

Validation built in
-------------------
The recorded en-XGB row and a recomputed 4-channel mean come from independent
draws of the sampling attack, so they will not be bit-identical. The script
reports their agreement first; if the 4-channel recomputation tracks the
recorded ensemble, the 3-channel recomputation is trustworthy by the same
mechanism.

Usage
-----
  python analyze_clean_channel.py "res/*/res_s1_adult__p*.txt"
  python analyze_clean_channel.py "res/2nd part/res_s1_active_adult__p2.txt"
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# 'None' is in pandas' default na_values, so defense=None parses as NaN. Every
# script in this repo has to undo that before grouping (see
# analyze_defense_results.py:139).
CONFIG = ['defense', 'defend_side', 'defend_scope', 'd_para', 'out_para']

# Three nested channel sets, one per defender capability. The ceiling on any
# defense is the attack error of the ensemble restricted to the channels that
# defense cannot touch.
#
#   victim acting alone   can perturb {b_output}          -> ceiling over CLEAN
#   victim + trusted 3rd  can perturb {b_output, b_grad}  -> ceiling over OWN
#   nobody                                                -> OWN is irreducible
#
# b_grad is computed BY the adversary and sent down, so perturbing it in transit
# needs a party the adversary does not control (ProVFL's own D1 assumption).
CLEAN = ['b_grad', 'a_grad', 'a_output']          # survives a victim-only forward defense
OWN = ['a_grad', 'a_output']                      # the adversary's purely local tensors
ALL4 = CLEAN + ['b_output']


def load(patterns):
    frames = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else [])
        if not hits:
            print('[warn] no file matches %r' % pat, file=sys.stderr)
        for path in hits:
            df = pd.read_csv(path, on_bad_lines='skip', skip_blank_lines=True)
            df = df[df['pred_frac'].astype(str) != 'pred_frac']
            df['src'] = os.path.basename(path)
            frames.append(df)
            print('[load] %-34s %5d rows' % (os.path.basename(path), len(df)),
                  file=sys.stderr)
    if not frames:
        sys.exit('no result rows found')
    df = pd.concat(frames, ignore_index=True, sort=False)

    df['defense'] = df['defense'].fillna('None').astype(str)
    for col in ('pred_frac', 'gnd_frac', 'auc', 'accuracy', 'seed',
                'd_para', 'out_para'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ('defend_side', 'defend_scope'):
        if col not in df.columns:
            df[col] = 'both'          # pre-Step-A rows behaved this way
        df[col] = df[col].fillna('both').astype(str)
    return df.dropna(subset=['pred_frac', 'gnd_frac'])


def label(row):
    """Compact, sortable name for one experimental configuration."""
    d = row['defense']
    if d == 'None':
        return 'None'
    strength = row['out_para'] if row['out_para'] >= 0 else row['d_para']
    if d in ('shuffle', 'withdraw'):
        return '%s %.1f' % (d, row['d_para'])
    return '%-10s s=%.3f %s/%s' % (d, strength, row['defend_side'],
                                   row['defend_scope'])


def build(df):
    """One row per (config, seed): the per-channel estimates side by side."""
    singles = df[df['attack_feat'].isin(ALL4)].copy()
    key = CONFIG + ['seed']
    wide = (singles.pivot_table(index=key, columns='attack_feat',
                                values='pred_frac', aggfunc='mean')
                   .reset_index())
    missing = [c for c in ALL4 if c not in wide.columns]
    if missing:
        sys.exit('result file has no %s rows -- cannot form the clean ensemble'
                 % ','.join(missing))

    truth = singles.groupby(key, dropna=False)['gnd_frac'].mean().rename('gnd_frac')
    aucs = singles.groupby(key, dropna=False)[['auc', 'accuracy']].mean()
    wide = wide.merge(truth, on=key).merge(aucs, on=key)

    wide['pred_clean'] = wide[CLEAN].mean(axis=1)
    wide['pred_own'] = wide[OWN].mean(axis=1)
    wide['pred_all4'] = wide[ALL4].mean(axis=1)

    recorded = df[df['attack_feat'] == 'all']
    if len(recorded):
        rec = recorded.groupby(key, dropna=False)['pred_frac'].mean().rename('pred_recorded')
        wide = wide.merge(rec, on=key, how='left')

    for name in ('clean', 'own', 'all4', 'recorded'):
        col = 'pred_%s' % name
        if col in wide.columns:
            wide['MAE_%s' % name] = (wide[col] - wide['gnd_frac']).abs()
    wide['MAE_b_output'] = (wide['b_output'] - wide['gnd_frac']).abs()
    return wide


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='result CSV/TXT paths or globs')
    ap.add_argument('--csv_out', default='', help='write the per-config table here')
    a = ap.parse_args(argv)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 40)

    wide = build(load(a.files))

    if 'MAE_recorded' in wide.columns and wide['MAE_recorded'].notna().any():
        ok = wide.dropna(subset=['MAE_recorded'])
        gap = (ok['pred_all4'] - ok['pred_recorded']).abs()
        corr = np.corrcoef(ok['pred_all4'], ok['pred_recorded'])[0, 1]
        print('\n=== validation: recomputed 4-channel mean vs the recorded en-XGB row ===')
        print('  n=%d   mean |diff| = %.4f   max |diff| = %.4f   r = %.4f'
              % (len(ok), gap.mean(), gap.max(), corr))
        print('  (independent sampling draws, so exact equality is not expected;'
              ' close agreement is what licenses the 3-channel recomputation)')

    grp = wide.groupby(CONFIG, dropna=False)
    tab = grp.agg(n=('seed', 'size'),
                  MAE_all=('MAE_all4', 'mean'),
                  MAE_clean=('MAE_clean', 'mean'),
                  sd_clean=('MAE_clean', 'std'),
                  MAE_own=('MAE_own', 'mean'),
                  MAE_b_output=('MAE_b_output', 'mean'),
                  auc=('auc', 'mean')).reset_index()
    tab['config'] = tab.apply(label, axis=1)
    tab['all_minus_clean'] = tab['MAE_all'] - tab['MAE_clean']

    cols = ['config', 'n', 'MAE_b_output', 'MAE_all', 'MAE_clean', 'MAE_own', 'auc']
    print('\n=== per-configuration: what survives when the adversary drops channels ===')
    print('  MAE_b_output = the defended channel alone (how well the mechanism works)')
    print('  MAE_all      = the 4-channel ensemble ProVFL actually reports')
    print('  MAE_clean    = ensemble minus b_output  -> ceiling for a victim acting alone')
    print('  MAE_own      = a_grad + a_output only   -> irreducible, no party can touch it')
    print(tab[cols].sort_values('MAE_b_output', ascending=False)
             .to_string(index=False, float_format=lambda v: '%.4f' % v))

    base = wide[wide['defense'] == 'None']
    print('\n=== the two ceilings ===')
    if len(base):
        print('  undefended, all four channels           MAE = %.4f  (n=%d)'
              % (base['MAE_all4'].mean(), len(base)))
        print('  undefended, b_output discarded          MAE = %.4f'
              % base['MAE_clean'].mean())
        print('  undefended, adversary-local only        MAE = %.4f'
              % base['MAE_own'].mean())

    # Restrict to what a victim can actually deploy alone: forward channel only,
    # or a structural change to its own data. Everything else needs a TTP.
    forward = tab[(tab['defense'] == 'None') |
                  (((tab['defend_side'] == 'output') |
                    tab['defense'].isin(['shuffle', 'withdraw'])) &
                   (tab['defend_scope'] != 'both'))]
    print('\n  --- victim acting alone (forward channel or own data) ---')
    print('  configs                                 n = %d' % len(forward))
    print('  MAE_b_output range                      %.4f .. %.4f  (%.1fx)'
          % (forward['MAE_b_output'].min(), forward['MAE_b_output'].max(),
             forward['MAE_b_output'].max() / max(forward['MAE_b_output'].min(), 1e-9)))
    print('  MAE_clean     range                     %.4f .. %.4f  (%.1fx)'
          % (forward['MAE_clean'].min(), forward['MAE_clean'].max(),
             forward['MAE_clean'].max() / max(forward['MAE_clean'].min(), 1e-9)))
    print('  MAE_all       range                     %.4f .. %.4f'
          % (forward['MAE_all'].min(), forward['MAE_all'].max()))
    print('\n  Reading: the mechanisms do work -- MAE_b_output moves by the factor'
          '\n  above. The combined adversary does not care, because MAE_clean is'
          '\n  pinned near its undefended value: dropping the defended channel costs'
          '\n  it nothing. MAE_all is bounded by MAE_clean, not by MAE_b_output.')

    ttp = tab[(tab['defend_side'].isin(['grad', 'both'])) &
              (tab['defend_scope'] != 'both')]
    if len(ttp):
        print('\n  --- with a trusted third party perturbing b_grad in transit ---')
        print('  MAE_clean     range                     %.4f .. %.4f'
              % (ttp['MAE_clean'].min(), ttp['MAE_clean'].max()))
        print('  MAE_own       range                     %.4f .. %.4f  <- still the floor'
              % (ttp['MAE_own'].min(), ttp['MAE_own'].max()))
        print('  Gradient-side noise DOES raise MAE_clean, because b_grad sits inside'
              '\n  the clean set. That is where roadmap Table 1 protection came from,'
              '\n  and it is exactly the assumption D1 smuggles in.')

    if a.csv_out:
        tab[cols + CONFIG].to_csv(a.csv_out, index=False)
        print('\nwrote %s' % a.csv_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
