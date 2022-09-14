[![PyPI version](https://badge.fury.io/py/abcd-pyhf.svg)](https://badge.fury.io/py/abcd-pyhf)

`pyhf`-based implementation of the ABCD method for background estimation

# Instructions for LXPLUS

The code can be run on LXPLUS by following the instructions below.

Create a virtual Python environment:
```bash
python3 -m venv abcd-pyhf
```

This will create a directory named `abcd-pyhf` that will contain all the necessary packages to run the code.
Activate the virtual Python environment with the following command:
```bash
source abcd-pyhf/bin/activate
```

You can shut down the virtual Python environment at any time with the Bash command `deactivate`.
You can later reactivate the environment by running the Bash command `source abcd-pyhf/bin/activate`.

To run the code, you will have to download the necessary packages in the Python environment (this only needs to be done once):
```bash
pip install --upgrade pip
pip install abcd-pyhf
```

You can then use the package in Python:
```python
from abcd_pyhf import ABCD
```
