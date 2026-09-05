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

What is left is genuinely environmental: fetch the UCI Adult files, confirm the
accelerator is on, and prove the defense dispatcher behaves before any GPU time is
spent. Nothing here trains a model.

    python kaggle_setup.py --dataset adult      # default
    python kaggle_setup.py --skip_test          # fetch data only
"""
import argparse
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile

# Only the tabular datasets the campaign actually uses. celeba is deliberately absent:
# it needs the image archive and no session in kaggle_run.py touches it.
#
# Each entry is (destination, [candidate URLs]). UCI moved Adult from the old
# /ml/machine-learning-databases/ tree to /static/public/2/, and serves the pair as a
# zip there; both are listed because which one resolves has changed more than once.
DATA_URLS = {
    'adult': [
        ('data/adult/adult.data',
         ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data']),
        ('data/adult/adult.test',
         ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test']),
    ],
    'bankmk': [
        ('data/bankmk/bank-full.csv', []),        # not auto-downloadable, see README
    ],
}

# Fallback: one zip holding adult.data and adult.test byte-identically, extracted when
# the direct files fail.
DATA_ZIPS = {'adult': ('https://archive.ics.uci.edu/static/public/2/adult.zip',
                       ['adult.data', 'adult.test'])}

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
        print('  MISS %-28s no download URL; place it there by hand' % rel_path)
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


def fetch_zip_fallback(dataset, targets, insecure=False):
    """Last resort: pull the dataset's zip and extract only the members we need."""
    spec = DATA_ZIPS.get(dataset)
    if not spec:
        return False
    url, members = spec
    tmp = os.path.join('data', dataset, '_%s.zip' % dataset)
    print('  trying the zip fallback: %s' % url)
    err = _download(url, tmp, insecure=insecure)
    if err is not None:
        print('  FAIL zip %s' % err)
        return False
    ok = True
    try:
        with zipfile.ZipFile(tmp) as z:
            names = set(z.namelist())
            for member in members:
                dest = os.path.join('data', dataset, os.path.basename(member))
                if member not in names:
                    print('  FAIL %-28s not in the zip (has %s)'
                          % (dest, ', '.join(sorted(names))))
                    ok = False
                    continue
                with z.open(member) as src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out)
                print('  got  %-28s %8.1f KB (from zip)'
                      % (dest, os.path.getsize(dest) / 1024.0))
    except zipfile.BadZipFile as exc:
        print('  FAIL zip is not readable: %s' % exc)
        ok = False
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)
    return ok and all(os.path.isfile(t) and os.path.getsize(t) > 0 for t in targets)


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


def check_package_resolution():
    """Prove the repo's `datasets` beat any preinstalled package of the same name, and
    that dataloader imports without dragging in the celeba image stack."""
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
        from datasets.adult import TRAIN_DATA_FILE
        print('  dataloader imports clean | adult data at %s' % TRAIN_DATA_FILE)
        ok = ok and os.path.isfile(TRAIN_DATA_FILE)
        if not os.path.isfile(TRAIN_DATA_FILE):
            print('  but that file does not exist yet   <-- WRONG')
    except Exception as exc:                       # noqa: BLE001 - report and continue
        print('  dataloader import failed: %s' % exc)
        ok = False
    return ok


def run_unit_test():
    """test_defense_func.py: 66 pure-tensor checks, no dataset, under a second on CPU.

    Worth the second. It is what catches a defense that silently no-ops -- which would
    otherwise show up as a PrivacyGain of exactly 0.0000 after 22 minutes of GPU time.
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
    p.add_argument('--dataset', default='adult', choices=sorted(DATA_URLS),
                   help='which dataset files to fetch')
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
    print('\n[3/5] %s data' % a.dataset)
    spec = DATA_URLS[a.dataset]
    data = all([fetch(rel, urls, insecure=a.insecure) for rel, urls in spec])
    if not data:
        data = fetch_zip_fallback(a.dataset, [rel for rel, _ in spec],
                                  insecure=a.insecure)
    if not data and not a.insecure:
        print('  if the errors above are all CERTIFICATE_VERIFY_FAILED, this host\'s CA '
              'bundle is stale rather than UCI being down; retry with --insecure')
    print('\n[4/5] package resolution')
    pkg = check_package_resolution()
    print('\n[5/5] defense dispatcher')
    tests = True if a.skip_test else run_unit_test()

    print('\n' + '=' * 68)
    for name, ok in [('dependencies', deps), ('gpu', gpu), ('data', data),
                     ('imports', pkg), ('unit tests', tests)]:
        print('  %-14s %s' % (name, 'ok' if ok else 'FAILED'))
    if all([deps, gpu, data, pkg, tests]):
        print('\nready. Next, in order:')
        print('  !python kaggle_run.py --session s1 --smoke')
        print('  !python kaggle_run.py --session s1 --part 1/2')
        print('  !python kaggle_run.py --session s1 --part 2/2   # next notebook run')
        return 0
    print('\nfix the FAILED rows above before starting the campaign; running anyway '
          'either crashes on the first job or produces numbers you cannot use.')
    return 1


if __name__ == '__main__':
    sys.exit(main())

