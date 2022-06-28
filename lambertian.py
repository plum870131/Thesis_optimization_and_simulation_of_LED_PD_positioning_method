import numpy as np
import matplotlib.pyplot as plt
# import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
from funcfile import *
sympy.init_printing()


power = 100
theta = np.linspace(-45,45,200)
dis = np.linspace(3,5,20)
thetaa, diss = np.meshgrid(theta,dis)
leng = thetaa.shape

psi = np.zeros(leng)
# print(thetaa.shape,diss.shape,psi.shape)
pd = np.ones(3)
led = np.array([lamb_order(10),1])



p1 = src2pdcurrent(thetaa,psi,diss,pd,led)


fig = plt.figure()
ax3d = plt.axes(projection="3d")

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(thetaa,diss,p1, cmap='plasma')
ax3d.set_title('Radiant Flux at different distance and angle')
ax3d.set_xlabel('LED出射角')
ax3d.set_ylabel('距離')
ax3d.set_zlabel('Relative radiant flux(%)')
plt.show()