#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 12:58:29 (ywatanabe)"

import re
from codecs import open
from os import path

from setuptools import find_packages, setup

################################################################################
PACKAGE_NAME = "ezdsp"
PACKAGES = find_packages(where="src")
DESCRIPTION = "Easy digital signal processing"
KEYWORDS = ["numpy" "torch", "pandas", "LFP", "EEG", "MEG"]
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]
################################################################################

root_dir = path.abspath(path.dirname(__file__))


def _requirements():
    return [
        name.rstrip()
        for name in open(path.join(root_dir, "requirements.txt")).readlines()
    ]


with open(path.join(root_dir, "src", PACKAGE_NAME, "__init__.py")) as f:
    init_text = f.read()
    version = re.search(
        r"__version__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    license = re.search(
        r"__license__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    author = re.search(r"__author__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(
        1
    )
    author_email = re.search(
        r"__author_email__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    url = re.search(r"__url__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)

assert version
assert license
assert author
assert author_email
assert url

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    package_dir={"": "src"},
    name=PACKAGE_NAME,
    packages=PACKAGES,
    version=version,
    license=license,
    install_requires=_requirements(),
    author=author,
    author_email=author_email,
    url=url,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    python_requires=">=3.6",
)
