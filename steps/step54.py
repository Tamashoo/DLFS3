if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero
from dezero import Variable, as_variable
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
from dezero.models import MLP
from dezero import DataLoader
import math
import time
from dezero import test_mode

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)