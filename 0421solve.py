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

def cart2sph(cart_v):#3x?
    cart = np.divide(cart_v, np.sqrt(np.sum(np.square(cart_v),axis=0).reshape((1,-1))))
    return np.array([   np.arccos(cart[2,:])   ,    np.divide( np.arctan(cart[1,:]), cart[0,:]) + (cart[0,:]<0)*(np.pi)   ])#2x?

def rotate_y_mul(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad] list 1x?
    rot = np.zeros((ang.size,3,3))
    rot[:,0,0] = np.cos(ang)
    rot[:,0,2] = np.sin(ang)
    rot[:,1,1] = np.ones((ang.size,))
    rot[:,2,0] = -np.sin(ang)
    rot[:,2,2] = np.cos(ang)
    # np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    # print(rot)
    return rot #是一個matrix
def rotate_z_mul(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad] list 1x?
    rot = np.zeros((ang.size,3,3))
    rot[:,0,0] = np.cos(ang)
    rot[:,0,1] = -np.sin(ang)
    rot[:,1,0] = np.sin(ang)
    rot[:,1,1] = np.cos(ang)
    rot[:,2,2] = np.ones((ang.size,))
    #rot = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
    # print(rot)
    return rot #是一個matrix

# set environment


pd_num = 7
pd_m = 2
pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
alpha = np.deg2rad(45)#傾角
beta = np.deg2rad(360/pd_num)#方位角
const = 0.35

ori_tar = np.deg2rad(np.array([[70,20]])).T #2x1
ori_tar_cart = ori_ang2cart(ori_tar)#3x1
tar_car_correct = ori_tar_cart

pd_ori_ang = np.stack( (alpha*np.ones(pd_num),(beta*np.arange(1,pd_num+1))),0 )#2x?
pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd

pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3



in_ang = ang_from_ori(pd_ori_ang,ori_tar)#pdx1
activated = np.arange(0,pd_num,1)[in_ang.reshape((pd_num,))<pd_view]
light = const * np.power(  np.cos(in_ang)  ,pd_m)#pdx1
light[np.delete(np.arange(0,pd_num,1),activated),:] = np.zeros((pd_num-activated.size,1))

ref_accu = np.argmin(in_ang[activated,:])#activated中的
other_accu = np.delete(np.arange(0,activated.size,1),ref_accu)#activated中的
ref_accu_glob = activated[ref_accu]
other_accu_glob = activated[other_accu]

phi_ref = in_ang[ref_accu_glob,:].reshape((1,1))#1x1
phi_other = in_ang[other_accu_glob,:]#other x 1
ratio_accu = np.divide(np.cos(phi_ref),np.cos(phi_other)) #other x 1

threshold = 0


# from now on calculate: 只會get到light,pd_m,pd_ori_ang,pd_ori_cart,pd_rot_mat資訊
def solve_for_1led(light,pd_m,pd_ori_ang,threshold):
    pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
    pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3

    filt_l = np.arange(0,pd_num,1)[(light>threshold).flatten()]
    light_filt = np.tile(light[filt_l,:],(1,1))
    filt= light_filt.size #other+1

    # 以下light只剩下數值夠大的
    ref = np.argmax(light_filt)
    ref_glob = np.array(range(pd_num))[filt_l][ref]
    other = np.delete(np.array(range(filt)),ref)# in filt_l
    other_glob = np.array(range(pd_num))[filt_l][other]

    data_ref = light_filt[ref,:].reshape((1,1))#1x1
    data_other = light_filt[other,:] #other x 1



    ratio = np.power(np.divide(data_ref, data_other),1/pd_m) #other x 1


    # n1 = (o2-ratio*o1)/np.sqrt(np.sum(np.square(o2-ratio*o1)))
    nor = np.tile(pd_ori_car[:,ref_glob] ,(1,1))- np.multiply(ratio.T, pd_ori_car[:,other_glob]).T #other x 3
    # check = np.inner(nor,ori_tar_cart.T)

    ref_nor_inother = np.argmax(data_other) #in data other
    #ref_nor_glob = filt_l[other][ref_nor_inother]

    tar_car_sol = np.cross( nor[np.delete(np.arange(0,other.size,1),ref_nor_inother),:] , nor[ref_nor_inother,:].reshape((1,-1)) ) #other-1 x 3
    tar_car_sol = np.divide(tar_car_sol, np.sqrt(np.sum(np.square(tar_car_sol),axis = 1)).reshape((-1,1)))
    tar_car_sol = np.stack((tar_car_sol,-tar_car_sol))
    tar_car_sol = tar_car_sol[np.inner(tar_car_sol, pd_ori_car[:,ref_glob])>0,:]#other-1 x 3
    tar_ori_sol = cart2sph(tar_car_sol.T).T#other-1 x 2
    
    return tar_car_sol, filt_l



# solve for answer: tar_car_sol #other-1 x 3
tar_car_sol, filt_l = solve_for_1led(light,pd_m,pd_ori_ang,threshold)
filt = filt_l.size
correct = np.isclose(np.inner(tar_car_sol,ori_tar_cart.T) , np.ones((filt-2,1)))
print(correct)






def plot_3d_solve_1led(tar_car_sol, filt_l, sample=100):

# generate circles for plots
    circle =  np.stack((in_ang[filt_l]* np.ones((filt,sample)),\
        np.tile(np.linspace(0,2*np.pi,sample),(filt,1))))# 2 x filt x sample
    circle_cart = ori_ang2cart(circle.reshape((2,filt*sample))).reshape((3,filt,sample))# 3 x filt x sample
    circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample
    # 3 x pd x sample -> pd x 3 x sample
    # pd x 3 x 3
    # pd x  3 x sample
    circle_stereo = stereo_3dto2d(circle_rot.transpose((1,0,2)).reshape((3,filt*sample))).reshape((2,filt,sample)).transpose((1,0,2))# pd x  2 x sample
    # pd x  2 x sample
    
    
    
    o_stereo = stereo_3dto2d(pd_ori_car[:,filt_l])




    # Generate plot
    fig = plt.figure(figsize=plt.figaspect(2.))
    fig.suptitle('PD and Stereographic Projection')
    
    ax = fig.add_subplot(211, projection='3d')
    ax.set_box_aspect(aspect = (1,1,0.5))
    # ax.set_aspect("auto")
    
    # draw sphere
    u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi/2,20))
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
    
    l = [[] for j in range(filt)]
    p = [[] for j in range(filt)]
    t = [[] for j in range(filt-2)]
    
    for i in range(filt):
        l[i], = ax.plot(circle_rot[i,0,:],circle_rot[i,1,:],circle_rot[i,2,:])
        p[i] = ax.scatter(pd_ori_car[0,i],pd_ori_car[1,i],pd_ori_car[2,i])
    for i in range(filt-2):
        t[i] = ax.scatter(tar_car_sol[i,0],tar_car_sol[i,1],tar_car_sol[i,2],marker='3',s=1000,c = 'indigo')
    a,b,c = ori_tar_cart
    t1 = ax.scatter(a,b,c,marker='x',s=100,c='k')
    
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
    
    #ax.legend([l1,l2,l3,l4,l5],['pd1','pd2','target orientation','solve from normal','solve from rotate'],bbox_to_anchor=(-0.5, 1.3), loc='upper left')
    
    
    
    ax = fig.add_subplot(212)
    
    ax.axis('equal')
    
    for i in range(filt):
        l[i], = ax.plot(circle_stereo[i,0,:],circle_stereo[i,1,:])
        p[i] = ax.scatter(o_stereo[0,i],o_stereo[1,i])
    tar_car_sol_ste = stereo_3dto2d(tar_car_sol.T).T
    for i in range(filt-2):
        t[i] = ax.scatter(tar_car_sol_ste[i,0],tar_car_sol_ste[i,1],marker='3',s=1000,c = 'indigo')
    a,b = stereo_3dto2d(ori_tar_cart)
    
    t1 = ax.scatter(a,b,marker='x',s=100,c='k')
    
    
    
    ax.grid(True)
    ax.set_title('Stereographic projection')
    
    
    
    # plt.show()

plot_3d_solve_1led(tar_car_sol, filt_l)


''' try to solve grom stereographic
# set target ori
ori_tar = np.deg2rad(np.array([[15,20]])).T

# set pd ori
pd_num = 5
pd_ori = np.deg2rad(     np.stack( (30*np.ones(pd_num),  (360/pd_num*np.arange(1,pd_num+1))  )     ,0 )    ) #[2x?]


# set ref and other: ref is the nearest one
ref = 4
other = np.delete(np.array(range(pd_num)), [ref])
pd_ref_ori = pd_ori[:,ref].reshape((2,1)) #[2x1]
pd_other_ori = pd_ori[:,other] #[2xother]


# set angle sample amount
sample = 50

# calculate measurement: cos_ratio: cosref/cosother (always>1)
cos_ratio = np.divide(np.cos(ang_from_ori(ori_tar,pd_ref_ori)) , np.cos(ang_from_ori(ori_tar,pd_other_ori))).T #[other,2]

# calculate pd angle wrt ref_pf: check minumum ang sum
ori_diff = ang_from_ori(pd_ref_ori,pd_other_ori).T #[otherx1]
# print(ori_diff,'dif')

#calculate ang_ref and ang_other
# ang_ref = arccos(r*cos(ang_other))
# ang_other = start from arccos(r*cos(other))
ang_possib_other = np.multiply( np.repeat([np.linspace(0,1,sample)],other.size,axis=0), np.pi/2-np.arccos(np.divide(1,cos_ratio))  ) +np.arccos(np.divide(1,cos_ratio)) #[otherx100]
ang_possib_ref = np.arccos(np.multiply(cos_ratio,np.cos(ang_possib_other)))
ang_start = np.argmax((ang_possib_other+ang_possib_ref)>ori_diff, axis =1).reshape((other.size,1))

# pd project to stereographic plot
stereo_cart_ref = pol2cart(stereo_sph2pol(pd_ref_ori)) #[2,1]
stereo_cart_other = pol2cart(stereo_sph2pol(pd_other_ori))#[2,other]
stereo_cart_tar = pol2cart(stereo_sph2pol(ori_tar))#[2,1]
# print(stereo_cart_other)



r_ref = np.sqrt(np.sum(np.square(\
            (pol2cart(stereo_sph2pol( np.concatenate( (  ((ang_possib_ref + pd_ref_ori[0,0]).reshape((1,other.size * sample))),\
                            (np.ones((1,other.size * sample))*pd_ref_ori[1,0])) )    ))).reshape((2,other.size,sample))\
            - stereo_cart_ref.reshape((2,1,1))    \
        ),axis=0) )  
sss = pol2cart(stereo_sph2pol(\
                np.stack(\
                    ((ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
                ).reshape(2,other.size*sample)))\
            .reshape(2,other.size,sample)
sss1 = pol2cart(stereo_sph2pol(\
                np.stack(\
                    ((-ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
                ).reshape(2,other.size*sample)))\
            .reshape(2,other.size,sample)
abbbb = np.sqrt(np.sum(np.square(\
            pol2cart(stereo_sph2pol(\
                np.stack(\
                    ((ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
                ).reshape(2,other.size*sample)))\
            .reshape(2,other.size,sample)\
            -stereo_cart_ref.reshape((2,1,1)) \
        ),axis=0))
abbbb1 = np.sqrt(np.sum(np.square(\
            pol2cart(stereo_sph2pol(\
                np.stack(\
                    ((-ang_possib_ref + pd_ref_ori[0,0]),np.ones((other.size , sample))*pd_ref_ori[1,0])\
                ).reshape(2,other.size*sample)))\
            .reshape(2,other.size,sample)\
            -stereo_cart_ref.reshape((2,1,1)) \
        ),axis=0))
r_other =   np.sqrt(np.sum(np.square(\
                pol2cart(stereo_sph2pol(\
                    np.concatenate\
                            (((pd_other_ori[0,:].reshape((4,1))+ang_possib_other).reshape((1,other.size*sample)),\
                            np.repeat(pd_other_ori[1,:],sample).reshape((1,-1))) )\
                    )).reshape((2,other.size,sample))\
                - stereo_cart_other.reshape((2,-1,1))\
            ),axis=0))

d_vector = stereo_cart_other.T-stereo_cart_ref.T #[otherx2]
d = np.sqrt(np.sum(np.square(d_vector),axis=1)).reshape((-1,1)) #[otherx1]
a = np.divide( np.square(d)+np.square(r_ref)-np.square(r_other) , 2*np.square(d))  #[other x sample]
mid =   stereo_cart_ref.reshape(2,1,1) + \
        np.multiply( \
            d_vector.T.reshape(2,-1,1) , np.repeat(a.reshape(1,other.size,-1),2,axis=0\
        )) #[2(xy) x other x sample]
curve =    np.tile(mid,(2,1,1,1))+\
        np.multiply(\
            np.stack((np.ones((2,other.size,sample)),-1*np.ones((2,other.size,sample)))),\
            np.tile(\
                np.multiply(\
                    np.tile(np.sqrt(np.square(r_ref)-np.square(a)), (2,1,1)),\
                    np.divide(d_vector,d).T.reshape((2,other.size,1))\
                )
            ,(2,1,1,1)) \
        )#[2(兩條) x 2(xy) x other x sample]


print(curve.shape,'ya')
# print( np.repeat(pd_other_ori[1,:].reshape((other.size,1)),sample,axis=1 ).shape  ,'hi')

print(stereo_cart_ref.shape)
print(stereo_cart_other.shape)
'''