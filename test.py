import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
from mpl_toolkits import mplot3d
from geneticalgorithm import geneticalgorithm as ga
sympy.init_printing()

from funcfile import *

from geneticalgorithm import geneticalgorithm as ga

a = (np.mgrid[-1.5:1.5:4j, -1.5:1.5:4j, 1:3:4j].reshape(-1,4*4*4))
print(a.shape)

threshold = 0.001

pd_num = 5
led_num = 7
pd_m = 3
led_m =3
#pd_num = 7
#pd_m = 3
pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
pd_alpha = np.deg2rad(35)#傾角
pd_beta = np.deg2rad(360/pd_num)#方位角


#led_num = 5
#led_m = 10
led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
led_alpha = np.deg2rad(45)#傾角
led_beta = np.deg2rad(360/led_num)#方位角

pd_area = 1
led_pt = 1
pd_saturate = 0.1
pd_respon = 1
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
testp_pos = (np.mgrid[-1.5:1.5:4j, -1.5:1.5:4j, 1:3:4j].reshape(-1,4*4*4)) # 3x?
# kpos = testp_pos.shape[1]
testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
# krot = testp_rot.shape[1]
# testp_pos = np.array([[0,1,1],[0,0,1],[0,-1,2]]).T # 3x?
kpos = testp_pos.shape[1]
# testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
krot = testp_rot.shape[1]

#(kpos,krot,led_num,3)  
glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
glob_led_ori = np.tile(global_testp_after_rot(led_ori_car,testp_rot), (kpos,1,1,1)).transpose((0,1,3,2))
#(kpos,krot,pd_num,3)  
glob_inv_pd_pos = testp_rot_matlist(testp_rot).transpose(0,2,1)
glob_inv_pd_pos = (np.tile(glob_inv_pd_pos@ pd_pos,\
                            (kpos,1,1,1))\
                    -np.tile(glob_inv_pd_pos@testp_pos\
                             ,(pd_num,1,1,1)).transpose(3,1,2,0)\
                    ).transpose(0,1,3,2)
# print(glob_inv_pd_pos)

# print(glob_inv_pd_pos)

# krot,kpos,led_num,pd_num
dis,in_ang,out_ang = interactive_btw_pdled(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car)


# 在view angle外的寫nan

in_ang_view = filter_view_angle(in_ang,pd_view)
out_ang_view = filter_view_angle(out_ang,led_view)


const = pd_respon * pd_area * led_pt * (led_num+1)/(2*np.pi)
light = const * np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
# light = np.divide(np.multiply( np.power(np.cos(in_ang_view),pd_m), np.power(np.cos(out_ang_view),led_m) ), np.power(dis,2) )
mask_light= np.isnan(light)
light[mask_light] = 0

# =============================================================================
# 這裡處理加上noise的部分
# =============================================================================

boltz = 1.380649 * 10**(-23)
temp_k = 300
bandwidth = 300
elec_charge = 1.60217663 * 10**(-19)
shunt = 50

noise = np.sqrt(4*temp_k*boltz*bandwidth/shunt\
          + 2*elec_charge*bandwidth*light\
              ) #+ 2*elec_charge*bandwidth*dark
# print(noise[0,0,0,0])

light_noise = light + noise
              
# print(np.isclose(light,light_noise))


# -------以下是硬體部分------------------




# filter掉訊號中小於threshold的部分：nan
# krot,kpos,led_num,pd_num
light_f = np.copy(light_noise)
light_f[light_f <= threshold] = np.nan
light_f[light_f >= pd_saturate] = pd_saturate





# =============================================================================
# 判斷特定LED是否有>=三個PD接收（才能判斷方位）
# =============================================================================



led_usable = np.sum(~np.isnan(light_f),axis=3)>2 #kp,kr,led,
pd_usable = np.sum(~np.isnan(light_f),axis =2 )>2#kp,kr,pd,
# pd_usable[2]=False
# 遮掉unusable
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
# mask: 遮掉ref1
maskled = np.full(light_led.shape, False)
maskled[\
        np.repeat(np.arange(kpos),krot*led_num),\
        np.tile(np.repeat(np.arange(krot),led_num),kpos),\
        np.tile(np.arange(led_num),kpos*krot),\
        ref1_led.flatten()] = True #kp,kr,ledu, pd
maskpd = np.full(light_pd.shape, False)
maskpd[np.repeat(np.arange(kpos),krot*pd_num),\
    np.tile(np.repeat(np.arange(krot),pd_num),kpos),\
    ref1_pd.flatten(),
    np.tile(np.arange(pd_num),kpos*krot),\
    ] = True #kp,kr,led, pdu
led_data_ref = light_led.copy()
led_data_ref .mask = (light_led .mask | ~maskled)
led_data_ref = np.sort(led_data_ref,axis=3)[:,:,:,0].reshape(kpos,krot,led_num,1)
led_data_other = light_led.copy()
led_data_other.mask = (light_led.mask | maskled)

# led_data_ref = light_led[maskled].reshape(kpos,krot,-1,1)#kp kr ledu 1
#led_data_other = light_led[~maskled].reshape(ledu,-1)# ledu other
pd_data_ref = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
pd_data_ref.mask = (light_pd.mask | ~maskpd)
pd_data_ref = np.sort(pd_data_ref,axis=2)[:,:,0,:].reshape(kpos,krot,1,pd_num)
pd_data_other = light_pd.copy()#light_pd[maskpd].reshape(1,-1)#1 pdu
pd_data_other.mask = (light_pd .mask | maskpd)
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
check_ratio_sum = np.sum(~np.isclose(ratio_led,check_ratio,equal_nan=True))
# print('-------------------------------------')
# print('False Ratio:',check_ratio_sum)

# =============================================================================
# 計算平面normal vector[ledu other 3]
# =============================================================================
#kpos x krot x ledu x other x 3
conf_led = np.tile(pd_ori_car.T,(kpos,krot,led_num,1,1))
conf_led_ref = np.sort( (np.ma.masked_array(conf_led,np.tile((light_pd.mask | ~maskled),(3,1,1,1,1)).transpose(1,2,3,4,0))),axis=3)[:,:,:,0,:].reshape(kpos,krot,led_num,1,3)
conf_led_other = np.ma.masked_array(conf_led,np.tile(led_data_other.mask,(3,1,1,1,1)).transpose(1,2,3,4,0))
nor_led = conf_led_ref - np.multiply(ratio_led.reshape(kpos,krot,led_num,-1,1),conf_led_other)

check_dot_led = np.sum(np.multiply(np.tile(testp_pos,(krot,led_num,pd_num,1,1)).transpose(4,0,1,2,3),nor_led),axis=4)
check_dot_led = np.ma.masked_invalid(check_dot_led)
#check_dot_led = np.sum(~(np.isclose(check_dot_led,np.zeros(led_data_other.shape))|np.isnan(check_dot_led)))
check_dot_led_sum = np.sum(~(np.isclose(check_dot_led,np.zeros(check_dot_led.shape))))
#print(check_dot_led_sum)


# kp kr l p 3
conf_pd = np.tile(led_ori_car,(kpos,krot,pd_num,1,1)).transpose(0,1,4,2,3) # kp kr l p
# kp kr 1 p 3
conf_pd_ref = np.sort( (np.ma.masked_array(conf_pd,np.tile((light_pd.mask | ~maskpd),(3,1,1,1,1)).transpose(1,2,3,4,0))),axis=2)[:,:,0,:,:].reshape(kpos,krot,1,-1,3)
# kp kr l p 3
conf_pd_other = np.ma.masked_array(conf_pd,np.tile(pd_data_other.mask,(3,1,1,1,1)).transpose(1,2,3,4,0))
# kp kr l p
nor_pd = conf_pd_ref - np.multiply(ratio_pd.reshape(kpos,krot,led_num,-1,1),conf_pd_other)
# 驗算dot
check_dot_pd = np.sum(np.multiply(glob_inv_pd_pos.reshape(kpos,krot,1,-1,3),nor_pd),axis=4)
check_dot_pd = np.ma.masked_invalid(check_dot_pd)
check_dot_pd_sum = np.sum(~(np.isclose(check_dot_pd,np.zeros(check_dot_pd.shape))))
#print(check_dot_led_sum)
# print('-------------------------------------')
# print('False normal vector from pd view:' ,check_dot_led_sum)
# print('False normal vector from led view:' ,check_dot_pd_sum)



# =============================================================================
# 取led_data_other強度最大者作為ref2_led，當cross的基準
# 並利用maskled2將data other分兩半
# => 計算cross
# =============================================================================
ref2_led = np.nanargmax(led_data_other, axis = 3)
ref2_pd = np.nanargmax(pd_data_other, axis = 2) #pdu,


maskled2 = np.full(light_led.shape, False)
maskled2[\
        np.repeat(np.arange(kpos),krot*led_num),\
        np.tile(np.repeat(np.arange(krot),led_num),kpos),\
        np.tile(np.arange(led_num),kpos*krot),\
        ref2_led.flatten()] = True #kp,kr,ledu, pd

maskpd2 = np.full(light_pd.shape, False)
maskpd2[np.repeat(np.arange(kpos),krot*pd_num),\
    np.tile(np.repeat(np.arange(krot),pd_num),kpos),\
    ref2_pd.flatten(),
    np.tile(np.arange(pd_num),kpos*krot),\
    ] = True #kp,kr,led, pdu

# 將normal vector分兩半

nor_led_ref = nor_led.copy()
nor_led_ref.mask = np.tile((light_led.mask| ~maskled2) ,(3,1,1,1,1)).transpose(1,2,3,4,0)#light_led True遮掉unusable, ~maskled2 True是除了ref2以外的 遮掉不是ref2的
nor_led_ref = np.sort(nor_led_ref,axis=3)[:,:,:,0,:].reshape(kpos,krot,led_num,1,3)
nor_led_other = nor_led.copy()
nor_led_other.mask = (nor_led.mask|np.tile(maskled2,(3,1,1,1,1)).transpose(1,2,3,4,0))#nor_led True遮掉unusable、ref1, maskled2 True遮掉ref2的


nor_pd_ref = nor_pd.copy()
nor_pd_ref.mask = np.tile((light_pd.mask| ~maskpd2) ,(3,1,1,1,1)).transpose(1,2,3,4,0)#light_led True遮掉unusable, ~maskled2 True是除了ref2以外的 遮掉不是ref2的
nor_pd_ref = np.sort(nor_pd_ref,axis=2)[:,:,0,:,:].reshape(kpos,krot,1,-1,3)
nor_pd_other = nor_pd.copy()
nor_pd_other.mask = (nor_pd.mask|np.tile(maskpd2,(3,1,1,1,1)).transpose(1,2,3,4,0))#nor_led True遮掉unusable、ref1, maskled2 True遮掉ref2的
# print(nor_pd_other)
# nor_pd_ref = nor_pd[maskpd2].reshape(1,-1,3) #1,pdu,3
# nor_pd_other = nor_pd[~maskpd2].reshape(-1,pdu,3) #led-2,pdu,3

# =============================================================================
# # 計算各平面交軸：cross vector
# =============================================================================
# kp kr l p 3
cross_led = np.ma.masked_array(np.cross(np.tile(nor_led_ref,(1,1,1,pd_num,1)),nor_led_other)\
                               ,nor_led_other.mask )#ledu,other-1,3
cross_led = np.divide(cross_led, np.tile(np.sqrt(np.sum(np.square(cross_led),axis=4)),(1,1,1,1,1)).transpose(1,2,3,4,0))#ledu,other-1,3
cross_led_mask = np.sum(np.multiply(conf_led_ref, cross_led),axis=4)<0 ## kp kr l p
cross_led = np.ma.masked_array(np.where(np.tile(cross_led_mask,(3,1,1,1,1)).transpose(1,2,3,4,0),-cross_led,cross_led),\
                               nor_led_other.mask)
# 驗算cross
check_cross_led = np.sum(np.multiply(cross_led,(np.tile( \
                                                        np.divide(testp_pos,\
                                                                  np.tile(np.sqrt(np.sum(np.square(testp_pos),axis=0)),(1,1))\
                                                                  ),\
                                                        (krot,led_num,pd_num,1,1)).transpose(4,0,1,2,3))\
                                     ),axis=4)#kp kr l p
check_cross_led = np.isclose(np.ma.masked_invalid(check_cross_led),np.ones(check_cross_led.shape))
check_cross_led_sum = np.sum(~check_cross_led)
cross_led_av = np.nanmean(cross_led,(2,3)) #kp kr 3

# kp kr l p 3
cross_pd = np.ma.masked_array(np.cross(np.tile(nor_pd_ref,(1,1,led_num,1,1)),nor_pd_other)\
                               ,nor_pd_other.mask )#ledu,other-1,3

cross_pd = np.divide(cross_pd, np.tile(np.sqrt(np.sum(np.square(cross_pd),axis=4)),(1,1,1,1,1)).transpose(1,2,3,4,0))#ledu,other-1,3

cross_pd_mask = np.sum(np.multiply(conf_pd_ref, cross_pd),axis=4)<0 ## kp kr l p
cross_pd = np.ma.masked_array(np.where(np.tile(cross_pd_mask,(3,1,1,1,1)).transpose(1,2,3,4,0),-cross_pd,cross_pd),\
                               nor_pd_other.mask)

# 驗算cross
check_cross_pd = np.sum(np.multiply(cross_pd,(np.tile( \
                                                        np.divide(glob_inv_pd_pos,\
                                                                  np.tile(np.sqrt(np.sum(np.square(glob_inv_pd_pos),axis=3)),(1,1,1,1)).transpose(1,2,3,0)\
                                                                  ),\
                                                        (led_num,1,1,1,1)).transpose(1,2,0,3,4))\
                                     ),axis=4)#kp kr l p
# print(check_cross_pd.transpose(0,1,3,2,4))
check_cross_pd = np.isclose(np.ma.masked_invalid(check_cross_pd),np.ones(check_cross_pd.shape))
check_cross_pd_sum = np.sum(~check_cross_pd)

# print('------------------------------------')
# print('False cross vector from pd view:' ,check_cross_led_sum)
# print('False cross vector from pd view:' ,check_cross_pd_sum)



# 答案求平均（忽略nan）
ori_sol_pd_coor = np.nanmean(cross_led,axis = (2,3)).filled(fill_value=np.nan) #kp kr 3,
ori_sol_led_coor = np.nanmean(cross_pd,axis = (2,3)).filled(fill_value=np.nan) #kp kr 3,
# print(ori_sol_pd_coor)


# 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
sol_in_ang = np.arccos(np.inner(ori_sol_pd_coor,pd_ori_car.T)) # kp kr pd,
sol_out_ang = np.arccos(np.inner(ori_sol_led_coor,led_ori_car.T)) #kp kr led,
# print(sol_out_ang)
sol_dis = np.sqrt(const * np.divide(np.multiply(\
                                                np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1,1,1)).transpose(1,2,0,3),\
                                                np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1,1,1)).transpose(1,2,3,0)\
                                                ),\
                                    light_f)) #kp kr l p
# print(np.tile(sol_in_ang,(led_num,1,1,1)).transpose(1,2,0,3),';;;;;;;')
# print(np.tile(sol_out_ang,(pd_num,1,1,1)).transpose(1,2,3,0),'---------')
# print(sol_dis)
check_dis = np.sqrt(np.sum(np.square(np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose(1,2,3,0,4)),axis=4))
check_dis = np.sum(~np.isclose(np.ma.masked_invalid(sol_dis),check_dis))
# print('------------------------------------')
# print('False dis:' ,check_dis)
# print('------------------------------------')
sol_dis_av = np.nanmean(sol_dis,axis = (2,3))#kp kr
# print()
error = (np.sum(np.square(np.multiply(cross_led_av,sol_dis_av.reshape(kpos,-1,1))-glob_led_pos[:,:,0,:]),axis=2))
error = error.filled(np.inf)
error[np.isclose(error,np.zeros(error.shape))] = np.nan 
error = np.sqrt(error)
error[np.isnan(error)]= 0

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