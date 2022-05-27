import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
from itertools import combinations
sympy.init_printing()


from funcfile import *
np.set_printoptions(precision=4,suppress=True)

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

testp_pos = np.array([[0,1,1],[0,0,1],[0,-1,2]]).T # 3x?
kpos = testp_pos.shape[1]
testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
krot = testp_rot.shape[1]

#(kpos,krot,led_num,3)  # kpos krot m 3
glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
glob_led_ori = np.tile(global_testp_after_rot(led_ori_car,testp_rot), (kpos,1,1,1)).transpose((0,1,3,2))

glob_inv_pd_pos = testp_rot_matlist(-testp_rot)
glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,(kpos,1,1,1))+np.tile(glob_inv_pd_pos@testp_pos,(pd_num,1,1,1)).transpose(3,1,2,0)).transpose(0,1,3,2)


# print(glob_inv_pd_pos)

# krot,kpos,led_num,pd_num
dis,in_ang,out_ang = interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car)


# 在view angle外的寫nan

in_ang_view = filter_view_angle(in_ang,pd_view)
out_ang_view = filter_view_angle(out_ang,led_view)


const = pd_area * led_pt * (led_num+1)/(2*np.pi)
light = const * np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
# light = np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
mask_light= np.isnan(light)
light[mask_light] = 0

# =============================================================================
# 這裡處理加上noise的部分
# =============================================================================



# -------以下是硬體部分------------------




# filter掉訊號中小於threshold的部分：nan
# krot,kpos,led_num,pd_num
light_f = np.copy(light)
light_f[light_f <= threshold] = np.nan


# =============================================================================
# 先處理單個sample point
# =============================================================================

# light_f = light_f.squeeze() #led pd



# =============================================================================
# 判斷特定LED是否有>=三個PD接收（才能判斷方位）
# =============================================================================



led_usable = np.sum(~np.isnan(light_f),axis=3)>2 #kp,kr,led,
pd_usable = np.sum(~np.isnan(light_f),axis =2 )>2#kp,kr,pd,
# pd_usable[2]=False
light_led = np.ma.masked_array(light_f,np.tile(~led_usable,(pd_num,1,1,1)).transpose(1,2,3,0)) #kp,kr,ledu, pd
light_pd = np.ma.masked_array(light_f, np.tile(~pd_usable,(led_num,1,1,1)).transpose(1,2,0,3))#.reshape(kpos,krot,led_num,-1) #kp,kr,led, pdu
# print(light_f,"---------")

# =============================================================================
# 取強度最大者作為ref1_led，建立平面的基準
# 並利用maskled將light_led分成ref和other
# => 計算ratio_led
# =============================================================================
ledu = led_usable.sum(axis=2)#kp,kr
pdu = pd_usable.sum(axis=2)#kp,kr

# print('Led, Pd usable amount: ',ledu,pdu)
ref1_led = np.nanargmax(light_led, axis = 3) #kp,kr,ledu,
ref1_pd = np.nanargmax(light_pd, axis = 2) #kp,kr,pdu,
# =============================================================================
# print(light_led)
# print(ref1_led)
# =============================================================================

# =============================================================================
# print(np.repeat(np.arange(kpos),krot*led_num),'haha')
# print(np.tile(np.repeat(np.arange(krot),led_num),kpos),'haha')
# print(np.tile(np.arange(led_num),kpos*krot),'haha')
# print(ref1_led.flatten(),'haha')
# =============================================================================
maskled = np.full(light_led.shape, False)
maskled[\
        np.repeat(np.arange(kpos),krot*led_num),\
        np.tile(np.repeat(np.arange(krot),led_num),kpos),\
        np.tile(np.arange(led_num),kpos*krot),\
        ref1_led.flatten()] = True #kp,kr,ledu, pd
#print(maskled)
maskpd = np.full(light_pd.shape, False)
maskpd[np.repeat(np.arange(kpos),krot*pd_num),\
    np.tile(np.repeat(np.arange(krot),pd_num),kpos),\
    ref1_pd.flatten(),
    np.tile(np.arange(pd_num),kpos*krot),\
    ] = True #kp,kr,led, pdu
led_data_ref = light_led.copy()
led_data_ref .mask = (led_data_ref .mask | ~maskled)
led_data_ref = np.sort(led_data_ref,axis=3)[:,:,:,0].reshape(kpos,krot,led_num,1)
led_data_other = light_led.copy()
led_data_other.mask = (led_data_other.mask | maskled)

# led_data_ref = light_led[maskled].reshape(kpos,krot,-1,1)#kp kr ledu 1
#led_data_other = light_led[~maskled].reshape(ledu,-1)# ledu other
pd_data_ref = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
pd_data_ref.mask = (pd_data_ref.mask | ~maskpd)
pd_data_ref = np.sort(pd_data_ref,axis=2)[:,:,0,:].reshape(kpos,krot,1,pd_num)
pd_data_other = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
pd_data_other.mask = (pd_data_other .mask | maskpd)
# =============================================================================
# print(light_pd,'-----------')
# print(maskpd)
# =============================================================================
#pd_data_other = light_pd[~maskpd].reshape(led_num-1,-1) #other, pdu
# ref/other
#ratio_led = np.power(np.ma.divide(led_data_ref, led_data_other),1/pd_m) #led_u x other
ratio_led = np.power(np.divide(led_data_ref, led_data_other),1/pd_m)
ratio_pd = np.power(np.divide(pd_data_ref, pd_data_other),1/led_m) #other, pdu
# in_ang  krot,kpos,led_num,pd_num

check_inang_ref = (np.sort(np.ma.masked_array(in_ang_view,(light_led.mask | ~maskled)),axis=3)[:,:,:,0].reshape(kpos,krot,-1,1))
check_inang_other = (np.ma.masked_array(in_ang_view,led_data_other.mask))
check_ratio = np.divide(np.cos(check_inang_ref),np.cos(check_inang_other))
i,j = 0,0
print(ratio_led[i,j])
print('=================')
print(check_ratio[i,j])
print(np.sum(~np.isclose(ratio_led,check_ratio,equal_nan=True)))
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
# print(ori_sol_pd_coor,ori_sol_led_coor)


# 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
sol_in_ang = np.arccos(np.inner(pd_ori_car.T,ori_sol_pd_coor)) # pd,
sol_out_ang = np.arccos(np.inner(led_ori_car.T,ori_sol_led_coor)) #led,

sol_dis = np.sqrt(const * np.divide(np.multiply(\
                              np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1)),np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1)).T\
                 ),light_f))


# =============================================================================
# circle =  np.stack((in_ang[filt_l]* np.ones((filt,sample)),\
#     np.tile(np.linspace(0,2*np.pi,sample),(filt,1))))# 2 x filt x sample
# circle_cart = ori_ang2cart(circle.reshape((2,filt*sample))).reshape((3,filt,sample))# 3 x filt x sample
# circle_rot = pd_rot_mat[filt_l,:,:] @ circle_cart.transpose((1,0,2)) # filt x  3 x sample
# # 3 x pd x sample -> pd x 3 x sample
# # pd x 3 x 3
# # pd x  3 x sample
# circle_stereo = stereo_3dto2d(circle_rot.transpose((1,0,2)).reshape((3,filt*sample))).reshape((2,filt,sample)).transpose((1,0,2))# pd x  2 x sample
# # pd x  2 x sample
# 
# 
# 
# o_stereo = stereo_3dto2d(pd_ori_car[:,filt_l])
# 
# =============================================================================


# =============================================================================
# 
# # Generate plot
# fig = plt.figure(figsize=plt.figaspect(2.))
# fig.suptitle('PD and Stereographic Projection')
# 
# ax = fig.add_subplot(211, projection='3d')
# ax.set_box_aspect(aspect = (1,1,1))
# # ax.set_aspect("auto")
# 
# # draw sphere
# u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
# x = 0.1*np.cos(u)*np.sin(v)
# y = 0.1*np.sin(u)*np.sin(v)
# z = 0.1*np.cos(v)
# ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# 
# arrow = 0.2*np.array([[1,0,0],[0,1,0],[0,0,1]]).T
# ax.quiver(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=0.1, color=["r",'g','b'])
# arrow = rotate_mat(testp_rot) @ arrow
# ax.quiver(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=0.1, color=["r",'g','b'])
# 
# # =============================================================================
# # l = [[] for j in range(filt)]
# # p = [[] for j in range(filt)]
# # t = [[] for j in range(filt-2)]
# # 
# # for i in range(filt):
# #     l[i], = ax.plot(circle_rot[i,0,:],circle_rot[i,1,:],circle_rot[i,2,:])
# #     p[i] = ax.scatter(pd_ori_car[0,i],pd_ori_car[1,i],pd_ori_car[2,i])
# # for i in range(filt-2):
# #     t[i] = ax.scatter(tar_car_sol[i,0],tar_car_sol[i,1],tar_car_sol[i,2],marker='3',s=1000,c = 'indigo')
# # a,b,c = ori_tar_cart
# # t1 = ax.scatter(a,b,c,marker='x',s=100,c='k')
# # =============================================================================
# 
# # ax3d.set_title('Radiant Flux at different distance and angle')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# 
# # =============================================================================
# # x, y, z = np.array([0,0,0])
# # u, v, w = np.array([0,0,1.5])
# # ax.quiver(x,y,z,u,v,w,arrow_length_ratio_led=0.1, color="black")
# # =============================================================================
# 
# 
# ax.grid(True)
# ax.set_xlim3d(-1.5,1.5)
# ax.set_ylim3d(-1.5,1.5)
# ax.set_zlim3d(0,3)
# 
# #ax.legend([l1,l2,l3,l4,l5],['pd1','pd2','target orientation','solve from nor_ledmal','solve from rotate'],bbox_to_anchor=(-0.5, 1.3), loc='upper left')
# 
# 
# 
# ax = fig.add_subplot(212)
# 
# ax.axis('equal')
# 
# # =============================================================================
# # for i in range(filt):
# #     l[i], = ax.plot(circle_stereo[i,0,:],circle_stereo[i,1,:])
# #     p[i] = ax.scatter(o_stereo[0,i],o_stereo[1,i])
# # tar_car_sol_ste = stereo_3dto2d(tar_car_sol.T).T
# # for i in range(filt-2):
# #     t[i] = ax.scatter(tar_car_sol_ste[i,0],tar_car_sol_ste[i,1],marker='3',s=1000,c = 'indigo')
# # a,b = stereo_3dto2d(ori_tar_cart)
# # 
# # t1 = ax.scatter(a,b,marker='x',s=100,c='k')
# # =============================================================================
# 
# 
# 
# ax.grid(True)
# ax.set_title('Stereographic projection')
# 
# 
# 
# # plt.show()
# =============================================================================





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















