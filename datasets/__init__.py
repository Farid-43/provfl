"""
Package marker. Without this file `datasets/` is only a namespace package, and Python
resolves namespace portions *after* it has scanned the whole of sys.path for a regular
package of the same name -- so on Kaggle, where HuggingFace `datasets` is preinstalled,
`from datasets.adult import ...` used to bind to site-packages and fail. The bootstrap
notebook worked around that with `pip uninstall -y datasets` plus renaming adult.py.

An empty __init__.py makes this directory a regular package, and regular packages are
found in sys.path order, so the repo checkout wins outright. Nothing is imported here
on purpose: dataloader.py imports each dataset module directly, and pulling them in
eagerly would drag torchvision and the CelebA files into every tabular run.
"""
