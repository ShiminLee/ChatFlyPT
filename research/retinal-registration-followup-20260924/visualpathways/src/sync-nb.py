#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os

""" Synchronize Jupyter notebooks with Python notebooks. """

for notebook_file in Path('.').rglob('*.ipynb'):
    if notebook_file.name.__contains__('-checkpoint'):
        continue
    print(notebook_file)
    os.system(f"jupytext --sync {notebook_file}")
