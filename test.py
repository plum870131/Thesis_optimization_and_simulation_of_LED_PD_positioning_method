import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

a = np.array([4,2,3,4,5,6,7,8])
print(np.multiply((a-3)>0,a))