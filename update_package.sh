#!/bin/bash

rm -rf build dist/* src/*.egg-info
tree src -I __pycache__ > tree.txt
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*

## EOF
