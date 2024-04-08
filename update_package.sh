#!/bin/bash

rm -rf build dist/* src/*.egg-info
tree src -I __pycache__ > tree.txt
tree example_outputs  >> tree.txt
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*

## EOF
