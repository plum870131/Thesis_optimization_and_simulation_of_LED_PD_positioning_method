import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

x = np.deg2rad(np.arange(0,90,0.5))
y = np.deg2rad(np.arange(0,90,1))
xx,yy = np.meshgrid(x,y)

# z = np.divide(np.tile(np.cos(x),(y.size,1)).T,np.tile(np.cos(y),(x.size,1)))
# x x y
# z=np.tile(np.cos(x),(y.size,1))
# z = np.tile(np.cos(y),(x.size,1)).T
z = np.divide(np.tile(np.cos(x),(y.size,1)),np.tile(np.cos(y),(x.size,1)).T)
print(xx.shape)
print(z.shape)

fig = plt.figure()
ax3d = plt.axes(projection="3d")

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(xx,yy,z, cmap='plasma')
# ax3d.set_title('Radiant Flux at different distance and angle')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
plt.show()