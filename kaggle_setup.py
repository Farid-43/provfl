#!/usr/bin/env python
"""
One-shot environment setup for the ProVFL campaign on Kaggle.

Run this once per notebook session, immediately after cloning the repo:

    !git clone -q https://github.com/Farid-43/provfl.git /kaggle/working/ProVFL
    %cd /kaggle/working/ProVFL
    !python kaggle_setup.py

It replaces the sed-based bootstrap that used to be pasted into the notebook. Those
patches are now fixes in the tree itself:

  * datasets/__init__.py makes the repo's `datasets` a regular package, so it wins over
    Kaggle's preinstalled HuggingFace `datasets` -- no `pip uninstall -y datasets`.
  * dataloader.py imports datasets.celeba lazily -- no sed on line 2.
  * datasets/adult.py resolves its data dir from the file's own location and uses
    pandas' .items() -- no path sed, no .iteritems() sed.
  * vfl_pia_active.py already casts prop_label.long() -- that sed was a no-op.

What is left is genuinely environmental: fetch the UCI data files, confirm the
accelerator is on, and prove the defense dispatcher behaves before any GPU time is
spent. Nothing here trains a model.

    python kaggle_setup.py --dataset adult      # default
    python kaggle_setup.py --dataset census     # before --session s1g on census
    python kaggle_setup.py --dataset bankmk     # before --session s1g on bankmk
    python kaggle_setup.py --skip_test          # fetch data only

census and bankmk are downloadable now (they were not before Round 7): UCI serves both
only as archives, census as a zip wrapping a tar.gz, so DATA_ARCHIVES walks one level
of nesting. Their loaders also used to hard-code '../data/...' and only worked when
launched from a subdirectory -- gate [4/5] now checks the resolved path per dataset.
"""
import argparse
import importlib
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

# Only the tabular datasets the campaign actually uses. celeba is deliberately absent:
# it needs the image archive and no session in kaggle_run.py touches it.
#
# Each entry is (destination, [candidate URLs]). UCI moved Adult from the old
# /ml/machine-learning-databases/ tree to /static/public/2/, and serves the pair as a
# zip there; both are listed because which one resolves has changed more than once.
#
# census and bankmk have no direct-file URLs at all any more -- UCI serves them only as
# archives, so their rows are empty lists and DATA_ARCHIVES below does the real work.
# The old census-income-mld/*.gz paths that the loader's comments implied are 404 as of
# 2026-09-06; verified, not assumed.
DATA_URLS = {
    'adult': [
        ('data/adult/adult.data',
         ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data']),
        ('data/adult/adult.test',
         ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test']),
    ],
    'bankmk': [
        ('data/bankmk/bank-full.csv', []),
    ],
    'census': [
        ('data/census/census-income.data', []),
        ('data/census/census-income.test', []),
    ],
}

# Archive fallbacks, tried in order when the direct files above fail or do not exist.
# `inner` handles one level of nesting: UCI's census download is a zip containing a
# single tar.gz containing the data, and its bank+marketing.zip is a zip of zips.
#
#   url      what to download
#   member   {name inside the archive: basename to write into data/<dataset>/}
#   inner    optional archive *inside* the outer one to descend into first
#
# Sizes for the record: bank.zip 0.6 MB; census+income+kdd.zip 9.8 MB expanding to
# ~156 MB (census-income.data is 103.9 MB). Both fine on Kaggle, neither fine to leave
# to a guess about member names -- these were read off the real archives.
DATA_ARCHIVES = {
    'adult': [
        dict(url='https://archive.ics.uci.edu/static/public/2/adult.zip',
             members={'adult.data': 'adult.data', 'adult.test': 'adult.test'}),
    ],
    'bankmk': [
        dict(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00222/'
                 'bank.zip',
             members={'bank-full.csv': 'bank-full.csv'}),
        dict(url='https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
             inner='bank.zip',
             members={'bank-full.csv': 'bank-full.csv'}),
    ],
    'census': [
        dict(url='https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip',
             inner='census.tar.gz',
             members={'census-income.data': 'census-income.data',
                      'census-income.test': 'census-income.test'}),
    ],
}

# What check_package_resolution should confirm is loadable, per dataset: the module that
# dataloader dispatches to, and a file that must exist for it to work.
DATASET_SENTINEL = {'adult': ('datasets.adult', 'data/adult/adult.data'),
                    'bankmk': ('datasets.bankmk', 'data/bankmk/bank-full.csv'),
                    'census': ('datasets.census', 'data/census/census-income.data')}

REQUIRED_IMPORTS = [('torch', 'torch'), ('pandas', 'pandas'), ('numpy', 'numpy'),
                    ('sklearn', 'scikit-learn'), ('xgboost', 'xgboost'),
                    ('scipy', 'scipy'), ('requests', 'requests')]


def _download(url, dest, insecure=False):
    """
    Fetch url to dest, returning an error string or None on success.

    Three transports are tried because they fail independently. urllib uses Python's
    own CA bundle, which on some hosts carries an expired root and rejects UCI with
    CERTIFICATE_VERIFY_FAILED while curl -- using the system store -- succeeds on the
    very same URL. --insecure skips verification and is never the default: a silently
    unverified download is how you end up training on a captive-portal error page.
    """
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    ctx = ssl._create_unverified_context() if insecure else None
    try:
        # UCI rejects the default urllib user agent
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
            body = r.read()
        if not body:
            raise ValueError('empty response')
        with open(dest, 'wb') as f:
            f.write(body)
        return None
    except Exception as exc:                       # noqa: BLE001 - try the next transport
        first = str(exc)

    for tool, cmd in (('curl', ['curl', '-sSfL', '--max-time', '180']
                               + (['-k'] if insecure else []) + ['-o', dest, url]),
                      ('wget', ['wget', '-q', '--timeout=180']
                               + (['--no-check-certificate'] if insecure else [])
                               + ['-O', dest, url])):
        if not shutil.which(tool):
            continue
        if subprocess.call(cmd) == 0 and os.path.isfile(dest) \
                and os.path.getsize(dest) > 0:
            return None
    return first


def fetch(rel_path, urls, insecure=False):
    """Ensure rel_path exists, trying each candidate URL in turn."""
    if os.path.isfile(rel_path) and os.path.getsize(rel_path) > 0:
        print('  have %-28s %8.1f KB' % (rel_path, os.path.getsize(rel_path) / 1024.0))
        return True
    if not urls:
        print('  need  %-28s no direct URL; the archive fallback will fetch it'
              % rel_path)
        return False
    err = 'no transport available'
    for url in urls:
        err = _download(url, rel_path, insecure=insecure)
        if err is None:
            print('  got  %-28s %8.1f KB'
                  % (rel_path, os.path.getsize(rel_path) / 1024.0))
            return True
    print('  FAIL %-28s %s' % (rel_path, err))
    return False


def _open_archive(path):
    """Return (kind, handle) for a zip or tar archive on disk."""
    if zipfile.is_zipfile(path):
        return 'zip', zipfile.ZipFile(path)
    return 'tar', tarfile.open(path, mode='r:*')


def _extract(members, dest_dir, names, reader):
    """Write each wanted member out. `names` lists what the archive holds, `reader`
    yields a file object for a given name. Returns True when all members landed."""
    ok = True
    for member, basename in sorted(members.items()):
        dest = os.path.join(dest_dir, basename)
        match = member if member in names else next(
            (n for n in names if os.path.basename(n) == member), None)
        if match is None:
            print('  FAIL %-28s not in the archive (has %s)'
                  % (dest, ', '.join(sorted(names)[:6])))
            ok = False
            continue
        with reader(match) as src, open(dest, 'wb') as out:
            shutil.copyfileobj(src, out)
        print('  got  %-28s %8.1f KB (extracted)'
              % (dest, os.path.getsize(dest) / 1024.0))
    return ok


def fetch_archive_fallback(dataset, targets, insecure=False):
    """Download the dataset's archive(s) and extract only the members we need.

    Handles one level of nesting because UCI needs it: census+income+kdd.zip holds a
    single census.tar.gz which holds the data, and bank+marketing.zip is a zip of zips.
    The inner archive is spooled to disk rather than held in memory -- census-income.data
    alone is 103.9 MB, and a Kaggle session has better uses for that RAM.
    """
    specs = DATA_ARCHIVES.get(dataset) or []
    dest_dir = os.path.join('data', dataset)
    os.makedirs(dest_dir, exist_ok=True)
    for spec in specs:
        tmp = os.path.join(dest_dir, '_outer_%s' % dataset)
        print('  trying the archive fallback: %s' % spec['url'])
        err = _download(spec['url'], tmp, insecure=insecure)
        if err is not None:
            print('  FAIL download: %s' % err)
            continue
        inner_tmp = None
        try:
            kind, arch = _open_archive(tmp)
            with arch:
                if kind == 'zip':
                    names, reader = arch.namelist(), arch.open
                else:
                    names = arch.getnames()
                    reader = lambda n: arch.extractfile(n)      # noqa: E731
                if spec.get('inner'):
                    match = next((n for n in names
                                  if os.path.basename(n) == spec['inner']), None)
                    if match is None:
                        print('  FAIL inner archive %s not present (has %s)'
                              % (spec['inner'], ', '.join(sorted(names)[:6])))
                        continue
                    inner_tmp = os.path.join(dest_dir, '_inner_%s' % dataset)
                    with reader(match) as src, open(inner_tmp, 'wb') as out:
                        shutil.copyfileobj(src, out)
                    kind2, inner = _open_archive(inner_tmp)
                    with inner:
                        if kind2 == 'zip':
                            got = _extract(spec['members'], dest_dir,
                                           inner.namelist(), inner.open)
                        else:
                            got = _extract(spec['members'], dest_dir,
                                           inner.getnames(), inner.extractfile)
                else:
                    got = _extract(spec['members'], dest_dir, names, reader)
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
            print('  FAIL archive is not readable: %s' % exc)
            got = False
        finally:
            for p in (tmp, inner_tmp):
                if p and os.path.isfile(p):
                    os.remove(p)
        if got and all(os.path.isfile(t) and os.path.getsize(t) > 0 for t in targets):
            return True
    return False


def check_imports():
    """Report what is missing rather than running `pip install -r requirement.txt`.

    A blanket install on Kaggle can pull a CPU wheel over the preinstalled CUDA torch
    and silently cost the campaign its GPU, so anything absent is named for a targeted
    install instead of upgraded in bulk.
    """
    missing = []
    for mod, pkg in REQUIRED_IMPORTS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print('  missing: %s' % ', '.join(missing))
        print('  install with: pip install %s' % ' '.join(missing))
    else:
        print('  all %d imports present' % len(REQUIRED_IMPORTS))
    return not missing
def check_device():
    """The single most expensive thing to get wrong: a session with the accelerator off
    trains on CPU without erroring, and the 22-34 min/job estimates stop applying."""
    try:
        import torch
    except ImportError:
        print('  torch not importable -- install it before running the campaign')
        return False
    if torch.cuda.is_available():
        print('  cuda  %s | torch %s | %d device(s)'
              % (torch.cuda.get_device_name(0), torch.__version__,
                 torch.cuda.device_count()))
        return True
    print('  NO GPU: torch.cuda.is_available() is False (torch %s).' % torch.__version__)
    print('  Kaggle: Notebook settings -> Accelerator -> GPU, then restart the session.')
    return False


def check_package_resolution(dataset='adult'):
    """Prove the repo's `datasets` beat any preinstalled package of the same name, that
    dataloader imports without dragging in the celeba image stack, and that the loader
    for *this* dataset resolves its data directory to a file that exists.

    That last part is not pedantry. Only adult.py resolved its paths from __file__;
    census.py and bankmk.py hard-coded '../data/...', which from the repo root points
    outside the checkout. They now share datasets.resolve_data_dir, and this gate is
    what proves it before a session is spent finding out.
    """
    ok = True
    try:
        import datasets
        where = os.path.abspath(os.path.dirname(datasets.__file__))
        local = where == os.path.abspath('datasets')
        print('  datasets -> %s%s' % (datasets.__file__, '' if local else '   <-- WRONG'))
        ok = ok and local
    except ImportError as exc:
        print('  datasets import failed: %s' % exc)
        return False
    try:
        import dataloader                                            # noqa: F401
        print('  dataloader imports clean')
    except Exception as exc:                       # noqa: BLE001 - report and continue
        print('  dataloader import failed: %s' % exc)
        return False

    mod_name, sentinel = DATASET_SENTINEL.get(dataset, (None, None))
    if mod_name is None:
        return ok
    try:
        importlib.import_module(mod_name)
    except Exception as exc:                       # noqa: BLE001 - report and continue
        print('  %s import failed: %s' % (mod_name, exc))
        return False
    if dataset == 'adult':
        from datasets.adult import TRAIN_DATA_FILE
        resolved = TRAIN_DATA_FILE
    else:
        from datasets import resolve_data_dir
        names = ('bankmk', 'bankmarketing') if dataset == 'bankmk' else (dataset,)
        resolved = os.path.join(
            resolve_data_dir(*names, sentinel=os.path.basename(sentinel)),
            os.path.basename(sentinel))
    exists = os.path.isfile(resolved)
    print('  %s -> %s%s' % (mod_name, resolved, '' if exists else '   <-- MISSING'))
    return ok and exists


def run_unit_test():
    """test_defense_func.py: ~100 pure-tensor checks, no dataset, under a second on CPU.

    Worth the second. It is what catches a defense that silently no-ops -- which would
    otherwise show up as a PrivacyGain of exactly 0.0000 after 12 minutes of GPU time.
    This is also the first place the norm-family checks execute: torch is not installed
    on the authoring machine, so they arrive here syntax-verified only.
    """
    if not os.path.isfile('test_defense_func.py'):
        print('  test_defense_func.py not found, skipping')
        return True
    proc = subprocess.run([sys.executable, 'test_defense_func.py'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode('utf-8', 'replace')
    tail = [l for l in out.splitlines() if 'checks failed' in l or l.startswith('FAIL')]
    print('  ' + ('\n  '.join(tail) if tail else out.strip()[-400:]))
    return proc.returncode == 0
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='adult', choices=sorted(DATA_URLS) + ['all'],
                   help='which dataset files to fetch ("all" fetches adult, bankmk, and census)')
    p.add_argument('--skip_test', action='store_true',
                   help='do not run test_defense_func.py')
    p.add_argument('--insecure', action='store_true',
                   help='skip TLS verification when downloading (last resort: use only '
                        'if every transport reports CERTIFICATE_VERIFY_FAILED)')
    a = p.parse_args(argv)

    print('cwd: %s' % os.getcwd())
    print('\n[1/5] dependencies')
    deps = check_imports()
    print('\n[2/5] accelerator')
    gpu = check_device()

    datasets_to_check = list(sorted(DATA_URLS.keys())) if a.dataset == 'all' else [a.dataset]
    all_data_ok = True
    all_pkg_ok = True

    for ds in datasets_to_check:
        print('\n[3/5] %s data' % ds)
        spec = DATA_URLS[ds]
        data = all([fetch(rel, urls, insecure=a.insecure) for rel, urls in spec])
        if not data:
            data = fetch_archive_fallback(ds, [rel for rel, _ in spec],
                                          insecure=a.insecure)
        if not data and not a.insecure:
            print('  if the errors above are all CERTIFICATE_VERIFY_FAILED, this host\'s CA '
                  'bundle is stale rather than UCI being down; retry with --insecure')
        all_data_ok = all_data_ok and data

        print('\n[4/5] package resolution for %s' % ds)
        pkg = check_package_resolution(ds)
        all_pkg_ok = all_pkg_ok and pkg

    print('\n[5/5] defense dispatcher')
    tests = True if a.skip_test else run_unit_test()

    print('\n' + '=' * 68)
    for name, ok in [('dependencies', deps), ('gpu', gpu), ('data', all_data_ok),
                     ('imports', all_pkg_ok), ('unit tests', tests)]:
        print('  %-14s %s' % (name, 'ok' if ok else 'FAILED'))
    if all([deps, gpu, all_data_ok, all_pkg_ok, tests]):
        print('\nready. Next, in order:')
        print('  !python kaggle_run.py --session s1c            # adult-sex confound + norm_align')
        print('  !python kaggle_run.py --session s1g --dataset adult --property race')
        print('  !python kaggle_run.py --session s1g --dataset bankmk --property marital')
        print('  !python kaggle_run.py --session s1g --dataset census --property sex')
        return 0
    print('\nfix the FAILED rows above before starting the campaign; running anyway '
          'either crashes on the first job or produces numbers you cannot use.')
    return 1


if __name__ == '__main__':
    sys.exit(main())

