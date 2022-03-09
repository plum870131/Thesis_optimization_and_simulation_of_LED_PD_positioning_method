import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

def positive(lst):
    return [i for i in range(len(lst)) if lst[i] > 0] or None

a = np.array([1,5,10,3,20])
b = 4.5
print(a[positive(a-b)])