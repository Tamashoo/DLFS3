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

def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + 2 * pad - kernel_size) // stride + 1

H, W = 4, 4
KH, KW = 3, 3
SH, SW = 1, 1
PH, PW = 1, 1

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)