[![PyPI version](https://badge.fury.io/py/abcd-pyhf.svg)](https://badge.fury.io/py/abcd-pyhf)

pyhf implementation of the ABCD method for background estimation

# Instructions for lxplus

The code can be run on lxplus by following the instruction given below.

First, set up your environment with default version of python 3: 
```bash
setupATLAS
lsetup "python pilot-default"
```

Then create a virtual python environment:
```bash
python3 -m venv env
```

This will create a directory named `env` which will contain all the necessary packages to run the code. 
Activate the virtual python environment with the following command:
```bash
source env/bin/activate
```

Any moment, you can shut down the virtual python environment with the bash command `deactivate`.
You can later reactivate the environment by setting up the necessary python version on lxplus and 
run the bash command `source env/bin/activate`.

To run the code, you will have to download the necessary packages in the python environment (this only need to be done once):
```bash
pip install numpy matplotlib pyhf[contrib] iminuit
```

You can then use the package in python (in the `src` folder):
```python
from abcd_pyhf import ABCD
```
