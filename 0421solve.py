import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
from itertools import combinations
sympy.init_printing()


from funcfile import *


def ang_from_ori(a_ang,b_ang):#[2xa] [2xb]
    # a_angg = np.transpose(np.array([a_ang.T]*b_ang.shape[1]),(1,0,2))
    a = np.tile(a_ang,(b_ang.shape[1],1,1)).transpose(1,2,0)
    # print(a_ang.shape,'a')
    # b_angg = np.array([b_ang.T]*a_ang.shape[0])
    b = np.tile(b_ang,(a_ang.shape[1],1,1)).transpose(1,0,2)
    # print(b_ang.shape,'a')
    out = np.arccos(\
            np.multiply(np.multiply(np.sin(a[0,:,:]),np.sin(b[0,:,:])), np.cos(a[1,:,:]-b[1,:,:]))+\
            np.multiply(np.cos(a[0,:,:]),np.cos(b[0,:,:])))
    # out = np.arccos(  np.multiply( np.multiply\
    #     (np.sin(a_angg[:,:,0]),np.sin(b_angg[:,:,0])), np.cos(a_angg[:,:,1]-b_angg[:,:,1]))  + \
    #     np.multiply(np.cos(a_angg[:,:,0]), np.cos(b_angg[:,:,0]))  )
    # print(out.shape,'out')
    return out #[axb]

# def ang_btw(a1,b1,a2,b2): #alpha1,beta1,alpha2,beta2
#     a1,b1,a2,b2 = np.deg2rad(np.array([a1,b1,a2,b2]))
#     return np.arccos(np.sin(a1)*np.sin(a2)*np.cos(b1-b2)+np.cos(a1)*np.cos(a2))

def stereo_sph2pol(ori):#ori[2x?]
    new = np.zeros(ori.shape)
    new[0,:] = np.divide(np.sin(ori[0,:]), 1+np.cos(ori[0,:]) )
    new[1,:] = ori[1,:]
    return new #[2x?]
def stereo_pol2sph(pol): #pol:R,ang
    out = np.zeros(pol.shape)
    out[0,:] = 2 * np.arctan(pol[0,:])
    out[1,:] = pol[1,:]
    return out                     
def stereo_3dto2d(p3d):#p3d[3x?]
    p2d = np.divide( p3d[:2,:] , (1+p3d[2,:]) )
    return p2d
def stereo_2dto3d(p2d): #p2d[2x?]
    out = np.stack( (2*p2d[0,:], 2*p2d[1,:], 1-np.sum(np.square(p2d),axis=0)) , 0 ) #[3x?]
    return np.divide(out, 1+ np.sum(np.square(p2d),axis=0) ) #[3x?]
def pol2cart(pol): #pol(R,ang) [2x?]
    return np.multiply(np.stack( ( np.cos(pol[1,:]), np.sin(pol[1,:])  ),0 ), pol[0,:] )

def rodrigue(k_vec1,ang): #k:[3]
    k_vec = k_vec1.reshape((3,))
    k = (1/np.sqrt(np.sum(np.square(k_vec))))*k_vec
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    I = np.eye(3)
    R = I + np.sin(ang)*K + (1-np.cos(ang))*(np.matmul(K,K)) #3x3
    return R #3x3
def rodrigue_1mul(k_vec1,ang): #k:[3]
    k_vec = k_vec1.reshape((3,))
    k = (1/np.sqrt(np.sum(np.square(k_vec))))*k_vec
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    I = np.eye(3)
    R = np.tile(I,(ang.size,1,1)) \
        + np.multiply(np.sin(ang).reshape(-1,1,1),np.tile(K,(ang.size,1,1))) \
        + np.multiply((1-np.cos(ang)).reshape(-1,1,1),np.tile((np.matmul(K,K)),(ang.size,1,1))) #angx3x3
    return R #angx3x3

def rodrigue_mulmul(k_vec,ang): #k:[sample,3] ang[sample,]
    k = np.multiply((1/np.sqrt(np.sum(np.square(k_vec),axis=1))).reshape((-1,1)),k_vec) #sample,
    K = np.zeros((ang.size,3,3))#np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]]) #sample,3,3
    K[:,0,1] = -k[:,2]
    K[:,0,2] = k[:,1]
    K[:,1,0] = k[:,2]
    K[:,1,2] = -k[:,0]
    K[:,2,0] = -k[:,1]
    K[:,2,1] = k[:,0]
    I = np.eye(3)
    R = np.tile(I,(ang.size,1,1)) \
        + np.multiply(np.sin(ang).reshape(-1,1,1),K) \
        + np.multiply((1-np.cos(ang)).reshape(-1,1,1),(np.matmul(K,K))) #angx3x3
    return R #samplex3x3

# =============================================================================
# # set target ori
# ori_tar = np.deg2rad(np.array([[15,20]])).T
# 
# # set pd ori
# pd_num = 5
# pd_ori = np.deg2rad(     np.stack( (30*np.ones(pd_num),  (360/pd_num*np.arange(1,pd_num+1))  )     ,0 )    ) #[2x?]
# 
# 
# # set ref and other: ref is the nearest one
# ref = 4
# other = np.delete(np.array(range(pd_num)), [ref])
# pd_ref_ori = pd_ori[:,ref].reshape((2,1)) #[2x1]
# pd_other_ori = pd_ori[:,other] #[2xother]
# 
# 
# # set angle sample amount
# sample = 50
# 
# # calculate measurement: cos_ratio: cosref/cosother (always>1)
# cos_ratio = np.divide(np.cos(ang_from_ori(ori_tar,pd_ref_ori)) , np.cos(ang_from_ori(ori_tar,pd_other_ori))).T #[other,2]
# 
# # calculate pd angle wrt ref_pf: check minumum ang sum
# ori_diff = ang_from_ori(pd_ref_ori,pd_other_ori).T #[otherx1]
# # print(ori_diff,'dif')
# 
# '''calculate ang_ref and ang_other'''
# # ang_ref = arccos(r*cos(ang_other))
# # ang_other = start from arccos(r*cos(other))
# ang_possib_other = np.multiply( np.repeat([np.linspace(0,1,sample)],other.size,axis=0), np.pi/2-np.arccos(np.divide(1,cos_ratio))  ) +np.arccos(np.divide(1,cos_ratio)) #[otherx100]
# ang_possib_ref = np.arccos(np.multiply(cos_ratio,np.cos(ang_possib_other)))
# ang_start = np.argmax((ang_possib_other+ang_possib_ref)>ori_diff, axis =1).reshape((other.size,1))
# 
# # pd project to stereographic plot
# stereo_cart_ref = pol2cart(stereo_sph2pol(pd_ref_ori)) #[2,1]
# stereo_cart_other = pol2cart(stereo_sph2pol(pd_other_ori))#[2,other]
# stereo_cart_tar = pol2cart(stereo_sph2pol(ori_tar))#[2,1]
# # print(stereo_cart_other)
# 
# 
# 
# r_ref = np.sqrt(np.sum(np.square(\
#             (pol2cart(stereo_sph2pol( np.concatenate( (  ((ang_possib_ref + pd_ref_ori[0,0]).reshape((1,other.size * sample))),\
#                            (np.ones((1,other.size * sample))*pd_ref_ori[1,0])) )    ))).reshape((2,other.size,sample))\
#             - stereo_cart_ref.reshape((2,1,1))    \
#         ),axis=0) )  
# sss = pol2cart(stereo_sph2pol(\
#                 np.stack(\
#                     ((ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
#                 ).reshape(2,other.size*sample)))\
#             .reshape(2,other.size,sample)
# sss1 = pol2cart(stereo_sph2pol(\
#                 np.stack(\
#                     ((-ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
#                 ).reshape(2,other.size*sample)))\
#             .reshape(2,other.size,sample)
# abbbb = np.sqrt(np.sum(np.square(\
#             pol2cart(stereo_sph2pol(\
#                 np.stack(\
#                     ((ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
#                 ).reshape(2,other.size*sample)))\
#             .reshape(2,other.size,sample)\
#             -stereo_cart_ref.reshape((2,1,1)) \
#         ),axis=0))
# abbbb1 = np.sqrt(np.sum(np.square(\
#             pol2cart(stereo_sph2pol(\
#                 np.stack(\
#                     ((-ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
#                 ).reshape(2,other.size*sample)))\
#             .reshape(2,other.size,sample)\
#             -stereo_cart_ref.reshape((2,1,1)) \
#         ),axis=0))
# =============================================================================





# =============================================================================
# r_other =   np.sqrt(np.sum(np.square(\
#                 pol2cart(stereo_sph2pol(\
#                     np.concatenate\
#                             (((pd_other_ori[0,:].reshape((4,1))+ang_possib_other).reshape((1,other.size*sample)),\
#                             np.repeat(pd_other_ori[1,:],sample).reshape((1,-1))) )\
#                     )).reshape((2,other.size,sample))\
#                 - stereo_cart_other.reshape((2,-1,1))\
#             ),axis=0))
# 
# d_vector = stereo_cart_other.T-stereo_cart_ref.T #[otherx2]
# d = np.sqrt(np.sum(np.square(d_vector),axis=1)).reshape((-1,1)) #[otherx1]
# a = np.divide( np.square(d)+np.square(r_ref)-np.square(r_other) , 2*np.square(d))  #[other x sample]
# mid =   stereo_cart_ref.reshape(2,1,1) + \
#         np.multiply( \
#             d_vector.T.reshape(2,-1,1) , np.repeat(a.reshape(1,other.size,-1),2,axis=0\
#         )) #[2(xy) x other x sample]
# curve =    np.tile(mid,(2,1,1,1))+\
#         np.multiply(\
#             np.stack((np.ones((2,other.size,sample)),-1*np.ones((2,other.size,sample)))),\
#             np.tile(\
#                 np.multiply(\
#                     np.tile(np.sqrt(np.square(r_ref)-np.square(a)), (2,1,1)),\
#                     np.divide(d_vector,d).T.reshape((2,other.size,1))\
#                 )
#             ,(2,1,1,1)) \
#         )#[2(兩條) x 2(xy) x other x sample]
# 
# 
# print(curve.shape,'ya')
# # print( np.repeat(pd_other_ori[1,:].reshape((other.size,1)),sample,axis=1 ).shape  ,'hi')
# 
# print(stereo_cart_ref.shape)
# print(stereo_cart_other.shape)
# 

sample = 50
ax1 = np.array([1,1,0])
ax2 = np.array([2,1,0])
a1 = np.deg2rad(30)
a2 = np.deg2rad(-20)
ori_tar = np.deg2rad(np.array([[50,150]])).T
ori_tar_cart = ori_ang2cart(ori_tar)

rot1 = rodrigue(ax1,a1)
o1 = np.matmul(rot1,np.array([0,0,1])).reshape((3,1))
o11 = (o1[0,:]<0)*(np.pi-2*np.arctan(o1[1,:]/o1[0,:]))
o1_ori = np.array([np.arccos(o1[2,:]),np.arctan(o1[1,:]/o1[0,:])+(o1[0,:]<0)*(np.pi)])
o1_new = np.array([np.sin(o1_ori[0,:])*np.cos(o1_ori[1,:]),np.sin(o1_ori[0,:])*np.sin(o1_ori[1,:]),np.cos(o1_ori[0,:])])
o1_f = stereo_3dto2d(o1)

rot2 = rodrigue(ax2,a2)
o2 = np.matmul(rot2,np.array([0,0,1])).reshape((3,1))
o2_f = stereo_3dto2d(o2)
o2_ori = np.array([np.arccos(o2[2,:]),np.arctan(o2[1,:]/o2[0,:])+(o2[0,:]<0)*(np.pi)])
o2_new = np.array([np.sin(o2_ori[0,:])*np.cos(o2_ori[1,:]),np.sin(o2_ori[0,:])*np.sin(o2_ori[1,:]),np.cos(o2_ori[0,:])])



ang_diff = ang_from_ori(o1_ori,o2_ori)
in_ang1 = ang_from_ori(ori_tar,o1_ori)
in_ang2 = ang_from_ori(ori_tar,o2_ori)
ratio = np.cos(in_ang2)/np.cos(in_ang1)

ang1 = np.linspace(np.arccos(1/ratio[0,0]),np.deg2rad(89),sample)
ang2 = np.arccos(ratio[0,0]*np.cos(ang1))


rr = np.sqrt(np.divide( np.square(np.cos(ang1))+np.square(np.cos(ang2))-2*np.multiply(np.multiply(np.cos(ang1),np.cos(ang2)),np.cos(ang_diff[0,0])) ,np.square(np.sin(ang_diff[0,0]))))
ang_side = np.arccos(rr) #[sample,]
ang_1to2 = np.arccos(np.cos(ang1)/rr)
axis_1to2 = np.cross(o1[:,0],o2[:,0])# 3,
mid = np.matmul(rodrigue_1mul(axis_1to2,ang_1to2),o1)#sample,3,1
axis_side = np.cross(mid.reshape((sample,3)),np.tile(axis_1to2,(sample,1)))   #  samplex3


p1 = np.matmul(rodrigue_mulmul(axis_side,ang_side),mid) #samplex3x1
p2 = np.matmul(rodrigue_mulmul(axis_side,-ang_side),mid)

check = np.divide(np.inner(np.squeeze(o2_new,axis=1),np.squeeze(p1,axis=2)),np.inner(np.squeeze(o1_new,axis=1),np.squeeze(p1,axis=2)))
check2 = np.divide(np.cos(ang2),np.cos(ang1))

p11= np.concatenate((p1[np.logical_not(np.isnan(p1[:,0,0])),:,0],p2[np.logical_not(np.isnan(p2[:,0,0])),:,0]))

circle1 = np.row_stack((  in_ang1*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
circle1_cart = ori_ang2cart(circle1)
# print(circle_cart)
# print(circle)
circle1_rot = np.matmul(rot1,circle1_cart)
circle1_flat = stereo_3dto2d(circle1_rot)


circle2 = np.row_stack((  in_ang2*np.ones((1,100))  ,np.linspace(0,2*np.pi,100)))
circle2_cart = ori_ang2cart(circle2)
# print(circle_cart)
# print(circle)
circle2_rot = np.matmul(rot2,circle2_cart)
circle2_flat = stereo_3dto2d(circle2_rot)
'''--------------------------------------------------------------------------------------------------'''

fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('PD and Stereographic Projection')

ax = fig.add_subplot(2, 1, 1, projection='3d')
ax.set_box_aspect(aspect = (1,1,0.5))
# ax.set_aspect("auto")

# draw sphere
u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi/2,20))
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

ax.plot(circle1_rot[0,:],circle1_rot[1,:],circle1_rot[2,:])
ax.plot(circle2_rot[0,:],circle2_rot[1,:],circle2_rot[2,:])

#ax.scatter(mid[:,0,:],mid[:,1,:],mid[:,2,:])
ax.scatter(p1[:,0,:],p1[:,1,:],p1[:,2,:],marker='x')
ax.scatter(p2[:,0,:],p2[:,1,:],p2[:,2,:],marker='x')

a,b,c = o1
ax.scatter(a,b,c,marker='x',c='r')
a,b,c = o2
ax.scatter(a,b,c,marker='x',c='b')
a,b,c = ori_tar_cart
ax.scatter(a,b,c,marker='x',s=100,c='k')

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



ax = fig.add_subplot(2, 1, 2)

ax.axis('equal')

ax.plot(circle1_flat[0,:],circle1_flat[1,:])
ax.scatter(o1_f[0,:],o1_f[1,:])
ax.plot(circle2_flat[0,:],circle2_flat[1,:])
ax.scatter(o2_f[0,:],o2_f[1,:])

a,b = stereo_3dto2d(np.squeeze(p1,axis=2).T)
ax.scatter(a,b,marker='x')
a,b = stereo_3dto2d(np.squeeze(p2,axis=2).T)
ax.scatter(a,b,marker='x')
a,b = stereo_3dto2d(ori_tar_cart)
ax.scatter(a,b,marker='x',c='k',s=100)

# =============================================================================
# ax.scatter(stereo_cart_ref[0],stereo_cart_ref[1],c='r',s =100)
# ax.scatter(stereo_cart_other[0],stereo_cart_other[1],c='b')
# ax.scatter(ori_tar[0],ori_tar[1],c='k')
# print(pd_ref_ori,'1')
# 
# 
# 
# ax.scatter(sss[0,:],sss[1,:],c='b')
# 
# ax.scatter(sss1[0,:],sss1[1,:],c='k')
# =============================================================================
# ax.plot(s2[0,:,:],s2[1,:,:])
# ax.plot(curve[0,0,:,:],curve[0,1,:,:])
# ax.plot(curve[1,0,:,:],curve[1,1,:,:])
#curve = [2(兩條) x 2(xy) x other x sample]
# ax.legend([1,2,3,4])
ax.grid(True)



# plt.show()