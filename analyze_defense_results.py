#!/usr/bin/env python
"""
Analyse ProVFL defense-experiment result CSVs.

The training scripts write one row per (attack_feat, classifier) evaluation via
my_utils/utils.write_to_csv, which dumps vars(args) plus the VFL metrics. Two
numbers the scripts do NOT store have to be derived here:

  MAE         = |pred_frac - gnd_frac|      the attack's error == the privacy metric
  PrivacyGain = MAE - MAE(defense='None')   the same number relative to no defense,
                                            computed inside each comparable cell

Older result files predate --defend_scope / --defend_side / --adaptive_noise /
--curriculum, so those columns are backfilled to the behaviour that produced them
(scope 'both', side 'both', no adaptive noise, no curriculum). Rows are also
tagged with the era they came from: vfl_pia_defense.py used to run at
val_ratio=0.2 with 5 seeds while vfl_pia_active.py ran at 0.3 with 10 seeds, so
pre- and post-unification rows are not numerically comparable. --compare warns
when a comparison straddles that boundary instead of silently mixing them.

Examples
--------
  python analyze_defense_results.py res_defense_adult.csv --group defense d_para
  python analyze_defense_results.py res_active_adult.csv \
      --filter attack_feat=all use_MR=1 use_LR=1 --group defense --pareto
  python analyze_defense_results.py "res_*.csv" --filter attack_feat=all \
      --compare "adaptive_noise=0" "adaptive_noise=1"
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except ImportError:          # scipy is in requirement.txt; degrade gracefully anyway
    _stats = None

# Columns identifying one comparable experimental cell. PrivacyGain is measured
# against the defense='None' rows that share these values.
CELL_COLS = ['dataset', 'property', 'attack_feat', 'classifier', 'use_MR', 'use_LR']

# Added after the first experiment batch; backfilled to what the old code did.
BACKFILL = {
    'defend_scope': 'both',      # the old loops perturbed a and b unconditionally
    'defend_side': 'both',       # and did so on outputs and gradients alike
    'adaptive_noise': 0,
    'curriculum': 0,
    'warmup_epochs': 0,
    'norm_threshold': 0.0,
    'sigma_low': np.nan,
    'sigma_high': np.nan,
    'sigma_alpha': np.nan,
    'use_MR': 0,                 # the passive script has no MR/LR knobs at all
    'use_LR': 0,
    'out_para': -1.0,
    'property': 'sex',
}

NUMERIC = ['pred_frac', 'gnd_frac', 'accuracy', 'auc', 'precision', 'recall',
           'used_time', 'd_para', 'd_para2', 'out_para', 'seed', 'use_MR', 'use_LR',
           'adaptive_noise', 'curriculum', 'warmup_epochs', 'norm_threshold',
           'sigma_low', 'sigma_high', 'sigma_alpha', 'epochs', 'attack_epoch',
           'sampling_size', 'val_ratio']

# A row is only usable if it carries the attack's prediction and the ground truth.
REQUIRED = ['pred_frac', 'gnd_frac', 'defense']

# The unification commit aligned both scripts on val_ratio=0.3 and 5 seeds. Files
# lacking a val_ratio column predate it; we tag them so comparisons stay honest.
LEGACY_VAL_RATIO = {'defense': 0.2, 'active': 0.3}

def _read_csv(path):
    """Tolerant CSV read: result files are appended to by many processes."""
    kw = dict(skip_blank_lines=True)
    try:
        return pd.read_csv(path, on_bad_lines='skip', **kw)
    except TypeError:                       # pandas < 1.3
        return pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, **kw)


def _script_of(path):
    """'defense' or 'active' from the filename, for the legacy val_ratio tag."""
    base = os.path.basename(path).lower()
    return 'active' if 'active' in base else 'defense'


def load_frame(patterns):
    """Concatenate every result file matching the given paths/globs."""
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits and os.path.exists(pat):
            hits = [pat]
        if not hits:
            print('[warn] no file matches %r' % pat, file=sys.stderr)
        paths.extend(hits)

    frames = []
    for path in paths:
        df = _read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            # e.g. res_baseline_adult.txt has DT_acc/NN_acc instead -- different script
            print('[skip] %s: no %s column' % (path, ','.join(missing)), file=sys.stderr)
            continue
        # write_to_csv appends headerless, so a header line can appear mid-file if
        # files were concatenated by hand
        df = df[df['pred_frac'].astype(str) != 'pred_frac']
        df['src_file'] = os.path.basename(path)
        df['src_script'] = _script_of(path)
        if 'val_ratio' not in df.columns:
            df['val_ratio'] = LEGACY_VAL_RATIO[_script_of(path)]
            df['era'] = 'legacy'
        else:
            df['era'] = 'unified'
        frames.append(df)
        print('[load] %-34s %5d rows' % (os.path.basename(path), len(df)), file=sys.stderr)

    if not frames:
        sys.exit('no usable result rows found')
    return pd.concat(frames, ignore_index=True, sort=False)

def prepare(df):
    """Backfill missing columns, coerce numerics, derive MAE and PrivacyGain."""
    for col, default in BACKFILL.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['defense'] = df['defense'].fillna('None').astype(str)
    df = df.dropna(subset=['pred_frac', 'gnd_frac']).copy()

    # The privacy metric: how far the attacker's estimated property fraction is
    # from the truth. Higher = better defended.
    df['MAE'] = (df['pred_frac'] - df['gnd_frac']).abs()

    cells = [c for c in CELL_COLS if c in df.columns]
    baseline = (df[df['defense'] == 'None']
                .groupby(cells, dropna=False)['MAE'].mean()
                .rename('MAE_none'))
    df = df.merge(baseline, how='left', left_on=cells, right_index=True)
    df['PrivacyGain'] = df['MAE'] - df['MAE_none']

    # Utility cost, same convention: negative = the defense hurt the model.
    acc_none = (df[df['defense'] == 'None']
                .groupby(cells, dropna=False)['auc'].mean()
                .rename('auc_none'))
    df = df.merge(acc_none, how='left', left_on=cells, right_index=True)
    df['AUCDrop'] = df['auc_none'] - df['auc']
    return df


def apply_filters(df, filters):
    """--filter col=val ... ; numeric columns compare numerically."""
    for spec in filters or []:
        if '=' not in spec:
            sys.exit('bad --filter %r, expected col=value' % spec)
        col, val = spec.split('=', 1)
        if col not in df.columns:
            sys.exit('unknown filter column %r (have: %s)'
                     % (col, ', '.join(sorted(df.columns))))
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            df = df[series == float(val)]
        else:
            df = df[series.astype(str) == val]
    return df

AGG_COLS = ['MAE', 'PrivacyGain', 'accuracy', 'auc', 'AUCDrop', 'used_time']


def summarise(df, group_cols):
    """Mean +/- std of the metrics, with the seed count behind each row."""
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        group_cols = ['defense']
    present = [c for c in AGG_COLS if c in df.columns]
    g = df.groupby(group_cols, dropna=False)
    out = g[present].agg(['mean', 'std'])
    out.columns = ['%s_%s' % (a, b) for a, b in out.columns]
    out['n'] = g.size()
    if 'seed' in df.columns:
        out['seeds'] = g['seed'].nunique()
    keep = ['n'] + (['seeds'] if 'seeds' in out.columns else [])
    for col in present:
        keep += ['%s_mean' % col, '%s_std' % col]
    return out[keep].sort_values('MAE_mean', ascending=False)


def compare(df, spec_a, spec_b, metric):
    """
    Paired comparison of two conditions, e.g. "adaptive_noise=0" vs "=1".

    Pairs on (seed, dataset, property, attack_feat, classifier, use_MR, use_LR) so
    the test sees matched runs. Falls back to an unpaired test when the conditions
    do not line up, and says so.
    """
    a = apply_filters(df.copy(), [spec_a])
    b = apply_filters(df.copy(), [spec_b])
    if a.empty or b.empty:
        print('  %-16s no rows for %s (n=%d) or %s (n=%d)'
              % (metric, spec_a, len(a), spec_b, len(b)))
        return

    pair_cols = [c for c in ['seed'] + CELL_COLS if c in df.columns]
    ja = a.groupby(pair_cols, dropna=False)[metric].mean()
    jb = b.groupby(pair_cols, dropna=False)[metric].mean()
    joined = pd.concat([ja.rename('a'), jb.rename('b')], axis=1).dropna()

    eras = set(a['era']).union(set(b['era']))
    if len(eras) > 1:
        print('  [warn] comparison mixes eras %s -- val_ratio/seed-count differ,'
              ' treat the p-value as indicative only' % sorted(eras))

    paired = len(joined) >= 2
    if paired:
        x, y = joined['a'].values, joined['b'].values
        test, npairs = 'paired t-test', len(joined)
        if _stats is None:
            stat = pval = float('nan')
        else:
            stat, pval = _stats.ttest_rel(y, x)
    else:
        x, y = a[metric].dropna().values, b[metric].dropna().values
        test, npairs = 'unpaired t-test (no matched pairs found)', min(len(x), len(y))
        if _stats is None or npairs < 2:
            stat = pval = float('nan')
        else:
            stat, pval = _stats.ttest_ind(y, x, equal_var=False)

    delta = float(np.mean(y) - np.mean(x))
    verdict = 'n/a' if np.isnan(pval) else ('SIGNIFICANT' if pval < 0.05 else 'not significant')
    print('  %-13s %-22s %8.4f  ->  %-22s %8.4f   delta %+8.4f'
          % (metric, spec_a, float(np.mean(x)), spec_b, float(np.mean(y)), delta))
    print('  %-13s %s, n=%d, t=%s, p=%s  [%s]'
          % ('', test, npairs,
             'nan' if np.isnan(stat) else '%.3f' % stat,
             'nan' if np.isnan(pval) else '%.4g' % pval, verdict))
    if _stats is None:
        print('  %-13s [warn] scipy unavailable -- no significance test run' % '')


def pareto_front(df, group_cols):
    """
    Non-dominated (MAE up, AUC up) points among the grouped means. This is the
    privacy-utility frontier: a defense is dominated if another gives at least as
    much attack error AND at least as much task AUC.
    """
    group_cols = [c for c in group_cols if c in df.columns] or ['defense']
    pts = (df.groupby(group_cols, dropna=False)[['MAE', 'auc', 'accuracy']]
             .mean().reset_index())
    keep = []
    for i, row in pts.iterrows():
        dominated = ((pts['MAE'] >= row['MAE']) & (pts['auc'] >= row['auc']) &
                     ((pts['MAE'] > row['MAE']) | (pts['auc'] > row['auc']))).any()
        if not dominated:
            keep.append(i)
    front = pts.loc[keep].sort_values('MAE', ascending=False)
    pts['on_front'] = pts.index.isin(keep)
    return front, pts.sort_values('MAE', ascending=False)

def build_parser():
    p = argparse.ArgumentParser(
        description='Summarise ProVFL defense results (MAE / PrivacyGain / Pareto).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+',
                   help='result CSV/TXT paths or globs (quote globs on Windows)')
    p.add_argument('--filter', nargs='*', default=[], metavar='COL=VAL',
                   help='restrict rows, e.g. --filter attack_feat=all use_MR=1')
    p.add_argument('--group', nargs='*', default=['defense'], metavar='COL',
                   help='grouping columns for the summary table (default: defense)')
    p.add_argument('--compare', nargs=2, metavar=('COND_A', 'COND_B'),
                   help='paired significance test between two conditions, '
                        'e.g. --compare "adaptive_noise=0" "adaptive_noise=1"')
    p.add_argument('--metrics', nargs='*', default=['MAE', 'PrivacyGain', 'auc', 'accuracy'],
                   help='metrics tested by --compare')
    p.add_argument('--pareto', action='store_true',
                   help='report the non-dominated (MAE, AUC) frontier')
    p.add_argument('--top', type=int, default=0, help='print only the top N summary rows')
    p.add_argument('--csv_out', type=str, default='',
                   help='also write the summary table to this CSV')
    p.add_argument('--list_columns', action='store_true',
                   help='print the available columns and exit')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.float_format', lambda v: '%.4f' % v)

    df = prepare(load_frame(args.files))
    if args.list_columns:
        print('\n'.join(sorted(df.columns)))
        return 0

    df = apply_filters(df, args.filter)
    if df.empty:
        sys.exit('no rows left after --filter')

    print('\n%d rows | %s | eras: %s'
          % (len(df), ', '.join(sorted(set(df['src_file']))), sorted(set(df['era']))))
    if df['MAE_none'].isna().any():
        n = int(df['MAE_none'].isna().sum())
        print('[warn] %d rows have no matching defense=None baseline in this selection'
              ' -- their PrivacyGain is NaN. Include the baseline runs to fix it.' % n)

    table = summarise(df, args.group)
    print('\n=== summary by %s ===' % ', '.join(args.group))
    print(table.head(args.top) if args.top else table)
    if args.csv_out:
        table.to_csv(args.csv_out)
        print('\nwrote %s' % args.csv_out)

    if args.pareto:
        front, allpts = pareto_front(df, args.group)
        print('\n=== privacy-utility frontier (MAE up, AUC up) ===')
        print(allpts.to_string(index=False))
        print('\non the frontier: %d of %d' % (len(front), len(allpts)))
        print(front.to_string(index=False))

    if args.compare:
        print('\n=== %s  vs  %s ===' % tuple(args.compare))
        for metric in args.metrics:
            if metric in df.columns:
                compare(df, args.compare[0], args.compare[1], metric)
            else:
                print('  [skip] no column %r' % metric)
    return 0


if __name__ == '__main__':
    sys.exit(main())
