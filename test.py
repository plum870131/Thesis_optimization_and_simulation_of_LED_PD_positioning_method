import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

a = np.zeros((2,4,3,5))
for i in range(2):
    for j in range(4):
        a[i,j,:,:] = a[i,j,:,:]+(4*i+j)
print(np.sqrt(np.multiply(a,a)))


