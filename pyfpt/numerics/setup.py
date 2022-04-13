#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:09:12 2021

@author: jjackson
"""

from setuptools import setup
from Cython.Build import cythonize

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}


setup(ext_modules=cythonize(["importance_sampling_sr_cython.pyx"],
                            **ext_options))
