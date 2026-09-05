"""
Package marker. Without this file `datasets/` is only a namespace package, and Python
resolves namespace portions *after* it has scanned the whole of sys.path for a regular
package of the same name -- so on Kaggle, where HuggingFace `datasets` is preinstalled,
`from datasets.adult import ...` used to bind to site-packages and fail. The bootstrap
notebook worked around that with `pip uninstall -y datasets` plus renaming adult.py.

An empty __init__.py makes this directory a regular package, and regular packages are
found in sys.path order, so the repo checkout wins outright. Nothing heavy is imported
here on purpose: dataloader.py imports each dataset module directly, and pulling them in
eagerly would drag torchvision and the CelebA files into every tabular run.

`resolve_data_dir` lives here because every loader needs it and none of them should have
to guess the process's working directory. adult.py keeps its own equivalent copy: that
path is verified end to end on Kaggle and is not worth re-touching to save eight lines.
"""
import os


def resolve_data_dir(*names, sentinel=None):
    """Locate the data directory for a dataset, independent of the working directory.

    The loaders used to hard-code '../data/<name>/', which only resolves when the
    process is launched from a subdirectory of the repo. On Kaggle the repo is cloned
    to /kaggle/working/ProVFL and the scripts run from there, so '../data' pointed
    outside the checkout entirely -- silently, until pandas raised FileNotFoundError
    partway into a session that had already been paid for.

    `names` are directory names to accept, in preference order, because the tree and
    the download script have disagreed about casing and abbreviation before
    ('bankmk' vs 'bankmarketing', 'Lawschool' vs 'lawschool'). `sentinel` is a file
    that must exist inside a candidate for it to count; without one the first
    directory that exists wins.

    Returns the first match, else the repo-local path built from `names[0]` so the
    resulting error names a path inside the checkout rather than one above it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    roots = [os.path.join(repo, 'data'),
             os.path.join(os.path.dirname(repo), 'data'),
             os.path.join(os.getcwd(), 'data')]
    env = os.environ.get('PROVFL_DATA', '')
    if env:
        roots = [env] + roots
    for root in roots:
        for name in names:
            d = os.path.join(root, name)
            if sentinel is None:
                if os.path.isdir(d):
                    return d
            elif os.path.isfile(os.path.join(d, sentinel)):
                return d
    return os.path.join(repo, 'data', names[0])

