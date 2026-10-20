import numpy as np
def read(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
