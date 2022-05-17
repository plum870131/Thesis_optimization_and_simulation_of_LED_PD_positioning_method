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

threshold = 0

pd_num = 7
pd_m = 3
pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
pd_alpha = np.deg2rad(35)#傾角
pd_beta = np.deg2rad(360/pd_num)#方位角


led_num = 5
led_m = 10
led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
led_alpha = np.deg2rad(45)#傾角
led_beta = np.deg2rad(360/led_num)#方位角

pd_area = 1
led_pt = 1

# =============================================================================
# ori_tar = np.deg2rad(np.array([[30,20]])).T #2x1
# ori_tar_cart = ori_ang2cart(ori_tar)#3x1
# tar_car_correct = ori_tar_cart
# =============================================================================

# config
pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd
pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3

led_pos = np.tile(np.array([[0,0,0]]).T,(1,led_num))
led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
led_ori_car = ori_ang2cart(led_ori_ang) #3xled
led_rot_mat = rotate_z_mul(led_ori_ang[1,:]) @ rotate_y_mul(led_ori_ang[0,:])#ledx3x3


# sample point

testp_pos = np.array([[1,1,1]]).T # 3x?
kpos = testp_pos.shape[1]
testp_rot = np.array([[np.pi,0,0]]).T
krot = testp_rot.shape[1]

#(kpos,krot,led_num,3)  # kpos krot m 3
glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
glob_led_ori = np.tile(global_testp_after_rot(led_ori_car,testp_rot), (kpos,1,1,1)).transpose((0,1,3,2))

glob_inv_pd_pos = testp_rot_matlist(-testp_rot)
glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,(kpos,1,1,1))+np.tile(glob_inv_pd_pos@testp_pos,(pd_num,1,1,1)).transpose(3,1,2,0)).transpose(0,1,3,2)


def interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car):
    (kpos,krot,led_num,_) = glob_led_pos.shape
    pd_num = pd_pos.shape[1]
    
    pos_delta = np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose((1,2,3,0,4)) \
        - np.tile(pd_pos.T,(kpos,krot,led_num,1,1))
    dis = np.sqrt(np.sum(np.square(pos_delta),axis=4)) # krot,kpos,led_num,pd_num
    #print(dis)
    in_ang = np.arccos(np.divide(np.sum(np.multiply( np.tile(pd_ori_car.T,(kpos,krot,led_num,1,1)), pos_delta), axis=4), dis))
    out_ang = np.arccos( np.divide(np.sum(np.multiply(    -pos_delta,np.tile(glob_led_ori,(pd_num,1,1,1,1)).transpose((1,2,3,0,4))   ),axis=4), dis ) )

    # krot,kpos,led_num,pd_num
    return dis,in_ang,out_ang

# krot,kpos,led_num,pd_num
dis,in_ang,out_ang = interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car)


# 在view angle外的寫nan
def filter_view_angle(mat,ang):
    mat_view = np.empty_like(mat)
    mat_view[:] = mat
    mat_view[mat_view >= ang] = np.nan
    return mat_view
in_ang_view = filter_view_angle(in_ang,pd_view)
out_ang_view = filter_view_angle(out_ang,led_view)


const = pd_area * led_pt * (led_num+1)/(2*np.pi)
light = const * np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
# light = np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
light[np.isnan(light)] = 0

# =============================================================================
# 這裡處理加上noise的部分
# =============================================================================



# -------以下是硬體部分------------------







# filter掉訊號中小於threshold的部分：nan
# krot,kpos,led_num,pd_num
light_f = np.empty_like(light)
light_f[light_f <= threshold] = np.nan


# =============================================================================
# 先處理單個sample point
# =============================================================================

light_f = light_f.squeeze() #led pd



# =============================================================================
# 判斷特定LED是否有>=三個PD接收（才能判斷方位）
# =============================================================================



led_usable = np.sum(~np.isnan(light_f),axis=1)>2 #led,
pd_usable = np.sum(~np.isnan(light_f),axis =0 )>2#pd,
pd_usable[2]=False
light_led = light_f[led_usable,:] #ledu, pd
light_pd = light_f[:,pd_usable] #led, pdu
# =============================================================================
# 取強度最大者作為ref1_led，建立平面的基準
# 並利用maskled將light_led分成ref和other
# => 計算ratio_led
# =============================================================================
ledu = led_usable.sum()
pdu = pd_usable.sum()

print('Led, Pd usable amount: ',ledu,pdu)
ref1_led = np.nanargmax(light_led, axis = 1) #ledu,
ref1_pd = np.nanargmax(light_pd, axis = 0) #pdu,
maskled = np.full(light_led.shape, False)
maskled[np.arange(ledu),ref1_led] = True #ledu, pd
maskpd = np.full(light_pd.shape, False)
maskpd[ref1_pd,np.arange(pdu)] = True #led, pdu

led_data_ref = light_led[maskled].reshape(-1,1)#ledu 1
led_data_other = light_led[~maskled].reshape(ledu,-1)# ledu other
pd_data_ref = light_pd[maskpd].reshape(1,-1)#1 pdu
pd_data_other = light_pd[~maskpd].reshape(led_num-1,-1) #other, pdu
# ref/other
ratio_led = np.power(np.divide(led_data_ref, led_data_other),1/pd_m) #led_u x other
ratio_pd = np.power(np.divide(pd_data_ref, pd_data_other),1/led_m) #other, pdu
# in_ang  krot,kpos,led_num,pd_num

# =============================================================================
# # ratio 驗算
# in_ang_t = in_ang_view.squeeze()
# in_ang_t = in_ang_t[led_usable,:]
# ratio_led_cor = np.divide(np.cos(in_ang_t[maskled].reshape(-1,1)),np.cos(in_ang_t[~maskled].reshape(ref1_led.sum(),-1)))
# out_ang_t = out_ang_view.squeeze()
# out_ang_t = out_ang_t[:,pd_usable]
# ratio_pd_cor = np.divide(np.cos(out_ang_t[maskpd].reshape(1,-1)),np.cos(out_ang_t[~maskpd].reshape(led_num-1,-1)))
# =============================================================================


# =============================================================================
# 計算平面normal vector[ledu other 3]
# =============================================================================
#ledu x other x 3
nor_led = np.tile(pd_ori_car.T,(ledu,1,1))[np.tile(maskled,(3,1,1)).transpose(1,2,0)].reshape(ledu,1,3)\
    - np.multiply(\
                np.tile(pd_ori_car.T,(ledu,1,1))[np.tile(~maskled,(3,1,1)).transpose(1,2,0)].reshape(ledu,-1,3)\
                ,ratio_led.reshape(ledu,-1,1))
#led-1 x pdu x 3
nor_pd = np.tile(led_ori_car,(pdu,1,1)).transpose(2,0,1)[np.tile(maskpd,(3,1,1)).transpose(1,2,0)].reshape(1,pdu,3)\
    - np.multiply(\
                np.tile(led_ori_car,(pdu,1,1)).transpose(2,0,1)[np.tile(~maskpd,(3,1,1)).transpose(1,2,0)].reshape(-1,pdu,3)\
                ,ratio_pd.reshape(-1,pdu,1))

# 驗算dot
check_dot_led = (np.inner(np.array(testp_pos.T),nor_led))
check_dot_led = np.sum(~(np.isclose(check_dot_led,np.zeros(led_data_other.shape))|np.isnan(check_dot_led)))
check_dot_pd = (np.inner(np.tile(glob_inv_pd_pos[0,0,0,:],(1,1,1)),nor_pd))[0,:,:]
check_dot_pd = np.sum(~(np.isclose(check_dot_pd,np.zeros(pd_data_other.shape))|np.isnan(check_dot_pd)))
print('-------------------------------------')
print('False normal vector from pd view:' ,check_dot_led)
print('False normal vector from led view:' ,check_dot_pd)
print('-------------------------------------')


# =============================================================================
# 取led_data_other強度最大者作為ref2_led，當cross的基準
# 並利用maskled2將data other分兩半
# => 計算cross
# =============================================================================
ref2_led = np.nanargmax(led_data_other, axis = 1)
ref2_pd = np.nanargmax(pd_data_other, axis = 0) #pdu,
maskled2 = np.full(led_data_other.shape, False)
maskled2[np.arange(ledu),ref2_led] = True #ledu, pd-1
maskpd2 = np.full(pd_data_other.shape, False)
maskpd2[ref2_pd, np.arange(pdu),] = True #led-1, pdu

# 將normal vector分兩半
nor_led_ref = nor_led[maskled2].reshape(-1,1,3) #ledu,1,3
nor_led_other = nor_led[~maskled2].reshape(ledu,-1,3) #ledu,other-1,3
nor_pd_ref = nor_pd[maskpd2].reshape(1,-1,3) #1,pdu,3
nor_pd_other = nor_pd[~maskpd2].reshape(-1,pdu,3) #led-2,pdu,3

# =============================================================================
# # 計算各平面交軸：cross vector
# =============================================================================
cross_led = np.cross(np.tile(nor_led_ref,(1,pd_num-2,1)),nor_led_other)#ledu,other-1,3
cross_led = np.divide(cross_led, np.tile(np.sqrt(np.sum(np.square(cross_led),axis=2)),(1,1,1)).transpose(1,2,0))#ledu,other-1,3
# cross_led_pstv：ref1的led指向，判斷軸方向是否正確
cross_led_pstv = np.tile(pd_ori_car.T,(ledu,1,1))  [np.tile(maskled,(3,1,1)).transpose(1,2,0)].reshape((-1,3)) #ledu 3
cross_led_mask = np.sum(np.multiply(cross_led, np.tile(cross_led_pstv,(pd_num-2,1,1)).transpose(1,0,2)),axis=2)<0#ledu other-1
cross_led = np.where(np.tile(cross_led_mask,(3,1,1)).transpose(1,2,0),-cross_led,cross_led)
#cross_led [ledu other-1 3]
cross_pd = np.cross(np.tile(nor_pd_ref,(led_num-2,1,1)),nor_pd_other) #led-2,pdu,3
cross_pd = np.divide(cross_pd, np.tile(np.sqrt(np.sum(np.square(cross_pd),axis=2)),(1,1,1)).transpose(1,2,0))#led-2,pdu,3
cross_pd_pstv = (np.tile(led_ori_car,(pdu,1,1)).transpose(2,0,1))
cross_pd_pstv = cross_pd_pstv [np.tile(maskpd,(3,1,1)).transpose(1,2,0)]    .reshape((1,-1,3)) #1,pdu 3
cross_pd_mask = np.sum(np.multiply(cross_pd, np.tile(cross_pd_pstv,(led_num-2,1,1))),axis=2)<0#led-2 pdu 3
cross_pd = np.where(np.tile(cross_pd_mask,(3,1,1)).transpose(1,2,0),-cross_pd,cross_pd)#led-2 pdu 3
#cross_pd [led-2 pdu 3]

# 驗算cross
check_cross_led = (np.sum(np.multiply(cross_led,np.tile(testp_pos.T/np.sqrt(np.sum(np.square(testp_pos))),(ledu,pd_num-2,1))),axis=2))
check_cross_led = np.sum(~(np.isnan(check_cross_led) | np.isclose(check_cross_led,np.ones((ledu,pd_num-2)))))
check_cross_pd = (np.sum(np.multiply(cross_pd,np.tile(-glob_inv_pd_pos[:,:,0,:]/np.sqrt(np.sum(np.square(glob_inv_pd_pos[:,:,0,:]))),(led_num-2,pd_usable.sum(),1))),axis=2))#led-2 pdu 3
check_cross_pd = np.sum(~(np.isnan(check_cross_pd) | np.isclose(check_cross_pd,np.ones((led_num-2,pdu)))))
print('------------------------------------')
print('False cross vector from pd view:' ,check_cross_led)
print('False cross vector from pd view:' ,check_cross_pd)
print('------------------------------------')


# 答案求平均（忽略nan）
ori_sol_pd_coor = np.nanmean(cross_led,axis = (0,1)) #3,
ori_sol_led_coor = np.nanmean(cross_pd,axis = (0,1)) #3,
print(ori_sol_pd_coor,ori_sol_led_coor)


# 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
sol_in_ang = np.arccos(np.inner(pd_ori_car.T,ori_sol_pd_coor)) # pd,
sol_out_ang = np.arccos(np.inner(led_ori_car.T,ori_sol_led_coor)) #led,

sol_dis = const * np.divide(np.multiply(\
                              np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1)),np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1)).T\
                 ),light_f)

print(sol_dis,dis)









'''










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
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio_led=0.1, color="black")
    ax.grid(False)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(0,1.5)
    
    #ax.legend([l1,l2,l3,l4,l5],['pd1','pd2','target orientation','solve from nor_ledmal','solve from rotate'],bbox_to_anchor=(-0.5, 1.3), loc='upper left')
    
    
    
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
'''















