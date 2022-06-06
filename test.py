import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
from mpl_toolkits import mplot3d
from geneticalgorithm import geneticalgorithm as ga
sympy.init_printing()

import funcfile as func

from geneticalgorithm import geneticalgorithm as ga

def f(X):
    return np.sum(X)


varbound=np.array([[0,10]]*3)

model=ga(function=f,dimension=3,variable_type='int',variable_boundaries=varbound)

model.run()


'''
def rodrigue(k,ang):
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    I = np.eye(3)
    R = I + np.sin(ang)*K + (1-np.cos(ang))*(np.matmul(K,K)) #3x3
    return R

o = [0,0]
# print(np.deg2rad(45)*np.ones((1,100)))
circle1 = np.row_stack((  np.deg2rad(45)*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
circle1_cart = func.ori_ang2cart(circle1)
rot1 = rodrigue(np.array([1,0,0]),np.deg2rad(30))
# print(circle_cart)
# print(circle)
circle1_rot = np.matmul(rot1,circle1_cart)
o1 = np.matmul(rot1,np.array([0,0,1]))

circle2 = np.row_stack((  np.deg2rad(20)*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
circle2_cart = func.ori_ang2cart(circle2)
rot2 = rodrigue(np.array([0,1,1]),np.deg2rad(38))
circle2_rot = np.matmul(rot2,circle2_cart)
o2 = np.matmul(rot2,np.array([0,0,1]))


# circle3 = np.row_stack((  np.deg2rad(20)*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
# circle3_cart = func.ori_ang2cart(circle3)
# o3 = np.array([0,0,1])
# k = []
# theta = np.deg2rad()






fig = plt.figure()
ax = fig.add_subplot( projection='3d')
ax.set_box_aspect(aspect = (1,1,0.5))
# ax.set_aspect("auto")

# draw sphere
u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi/2,20))
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

a,b,c = circle1_rot
ax.plot(a,b,c,c='r')
ax.scatter(o1[0],o1[1],o1[2],marker='x',c='r')
a,b,c = circle2_rot
ax.plot(a,b,c,c='g')
ax.scatter(o2[0],o2[1],o2[2],marker='x',c='g')
# a,b,c = circle3_cart
# ax.plot(a,b,c,c='b')
# ax.scatter(o3[0],o3[1],o3[2],marker='x',c='b')


# ax3d.set_title('Radiant Flux at different distance and angle')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x, y, z = np.array([0,0,0])
u, v, w = np.array([0,0,1.5])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
ax.grid(False)
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(0,1.5)


plt.show()

# power = 100
# theta = np.linspace(-45,45,200)
# dis = np.linspace(3,5,20)
# thetaa, diss = np.meshgrid(theta,dis)
# leng = thetaa.shape

# psi = np.zeros(leng)
# # print(thetaa.shape,diss.shape,psi.shape)
# pd = np.ones(3)
# led = np.array([func.lamb_order(10),1])

'''


'''
x = np.linspace(0,80,100)
# y = np.linspace(0,45,100)
# xx,yy = np.meshgrid(cos1,cos2)
# ratio = np.linspace(0.1,10,100)



ratio = 0.5
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)


ratio = 1.1
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)

ratio = 1.5
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)

ratio = 2
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)


ratio = 3
y = ratio*np.cos(np.deg2rad(x))
key = (y < 1.0).nonzero()
f = np.rad2deg(np.arccos(y[key]))
plt.plot(x[key],f)

plt.xlabel("x(degree)")
plt.ylabel("y(degree)")
plt.title('cosx/cosy對應角度')
plt.legend(['0.5','1.1','1.5','2','3'])
# plt.colorbar()

# fig = plt.figure()
# ax3d = plt.axes(projection="3d")

# ax3d = plt.axes(projection='3d')
# # ax3d.plot_surface(xx,yy,p1, cmap='plasma')
# plt.contourf(xx,yy,p1)
# ax3d.set_title('Radiant Flux at different distance and angle')
# ax3d.set_xlabel('x')
# ax3d.set_ylabel('y')
# ax3d.set_zlabel('Relative radiant flux(%)')
plt.show()
'''