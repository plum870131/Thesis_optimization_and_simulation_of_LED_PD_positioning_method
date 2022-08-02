#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:56:27 2022

@author: tiffany
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 05:27:56 2022

@author: tiffany
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:43:08 2022

@author: tiffany
"""
from funcfile import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.cm as cm
import matplotlib.colors as colors






def solve_mulmul():
# set environment

    global threshold #= 0.001
    
    global pd_num #= int(pd_num)
    global led_num# = int(led_num)
    global pd_m #= int(pd_m)
    global led_m #= int(led_m)
    

    pd_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    #led_num = 5
    #led_m = 10
    led_view = 2*np.arccos(np.exp(-np.log(2)/pd_m))
    # led_alpha = np.deg2rad(45)#傾角
    # led_beta = np.deg2rad(360/led_num)#方位角
    
    global pd_area #= 1
    global led_pt #= 1
    global pd_saturate #= np.inf
    global pd_respon# = 1
    
    # config
    pd_pos = np.tile(np.array([[0,0,0]]),(pd_num,1)).T # 3xpd_num
    global pd_ori_ang #= np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
    global pd_ori_car #= ori_ang2cart(pd_ori_ang) #3xpd
    pd_rot_mat = rotate_z_mul(pd_ori_ang[1,:]) @ rotate_y_mul(pd_ori_ang[0,:])#pdx3x3
    
    led_pos = np.tile(np.array([[0,0,0]]).T,(1,led_num))
    global led_ori_ang #= np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
    global led_ori_car #= ori_ang2cart(led_ori_ang) #3xled
    led_rot_mat = rotate_z_mul(led_ori_ang[1,:]) @ rotate_y_mul(led_ori_ang[0,:])#ledx3x3
    
    
    # sample point
    global testp_pos# = (np.mgrid[-1:1:4j, -1:1:4j, 1:3:4j].reshape(-1,4*4*4)) # 3x?
    # kpos = testp_pos.shape[1]
    global testp_rot #= np.array([[np.pi,0,0],[0,np.pi,0]]).T
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
    in_ang_view[in_ang_view>np.pi/2]=np.nan
    out_ang_view[out_ang_view>np.pi/2]=np.nan
    
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
    elec_charge = 1.60217663 * 10**(-19)
    
    global bandwidth #= 300
    global shunt #= 50
    global back_ground,dark_current,NEP

    thermal_noise = 4*temp_k*boltz*bandwidth/shunt
    
    noise = 1*np.sqrt(thermal_noise\
              + 2*elec_charge*bandwidth*(light+background+dark_current)\
                  ) #+ 2*elec_charge*bandwidth*dark
    # print(noise[0,0,0,0])
    light_noise = light + noise
    light_floor = NEP*np.floor_divide(light_noise, NEP)
    
    
    # print(np.nanmax(light_floor),'!!!!!')
    # -------以下是硬體部分------------------
    
    
    # snr = np.divide(noise,light_floor)
    # print(snr)
    
    # filter掉訊號中小於threshold的部分：nan
    # krot,kpos,led_num,pd_num
    light_f = np.copy(light_floor)
    light_f[light_f <= threshold] = np.nan
    light_f[light_f >= pd_saturate] = np.nan
    
    
    
    
    
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
    
    

    nor_led,nor_pd,conf_led_ref,conf_pd_ref,led_data_other,pd_data_other = get_surface(light_led,light_pd,led_num,pd_num,kpos,krot,led_m,pd_m,led_ori_car,pd_ori_car)
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
# =============================================================================
#     # 驗算cross
#     check_cross_led = np.sum(np.multiply(cross_led,(np.tile( \
#                                                             np.divide(testp_pos,\
#                                                                       np.tile(np.sqrt(np.sum(np.square(testp_pos),axis=0)),(1,1))\
#                                                                       ),\
#                                                             (krot,led_num,pd_num,1,1)).transpose(4,0,1,2,3))\
#                                          ),axis=4)#kp kr l p
#     check_cross_led = np.isclose(np.ma.masked_invalid(check_cross_led),np.ones(check_cross_led.shape))
#     check_cross_led_sum = np.sum(~check_cross_led)
# =============================================================================
    cross_led_av = np.nanmean(cross_led,(2,3)) #kp kr 3
    
    # cross_led_av = 
    
    # kp kr l p 3
    cross_pd = np.ma.masked_array(np.cross(np.tile(nor_pd_ref,(1,1,led_num,1,1)),nor_pd_other)\
                                   ,nor_pd_other.mask )#ledu,other-1,3
    
    cross_pd = np.divide(cross_pd, np.tile(np.sqrt(np.sum(np.square(cross_pd),axis=4)),(1,1,1,1,1)).transpose(1,2,3,4,0))#ledu,other-1,3
    
    cross_pd_mask = np.sum(np.multiply(conf_pd_ref, cross_pd),axis=4)<0 ## kp kr l p
    cross_pd = np.ma.masked_array(np.where(np.tile(cross_pd_mask,(3,1,1,1,1)).transpose(1,2,3,4,0),-cross_pd,cross_pd),\
                                   nor_pd_other.mask)
    
# =============================================================================
#     # 驗算cross
#     check_cross_pd = np.sum(np.multiply(cross_pd,(np.tile( \
#                                                             np.divide(glob_inv_pd_pos,\
#                                                                       np.tile(np.sqrt(np.sum(np.square(glob_inv_pd_pos),axis=3)),(1,1,1,1)).transpose(1,2,3,0)\
#                                                                       ),\
#                                                             (led_num,1,1,1,1)).transpose(1,2,0,3,4))\
#                                          ),axis=4)#kp kr l p
#     # print(check_cross_pd.transpose(0,1,3,2,4))
#     check_cross_pd = np.isclose(np.ma.masked_invalid(check_cross_pd),np.ones(check_cross_pd.shape))
#     check_cross_pd_sum = np.sum(~check_cross_pd)
# =============================================================================
    
    # print('------------------------------------')
    # print('False cross vector from pd view:' ,check_cross_led_sum)
    # print('False cross vector from pd view:' ,check_cross_pd_sum)
    
    
    # weight_form = 'mean''weight'
    
    global weight_form 
    mask_led = ~np.isnan(cross_led[:,:,:,:,0].filled(fill_value=np.nan))
    mask_pd = ~np.isnan(cross_pd[:,:,:,:,0].filled(fill_value=np.nan))
    mask_total = (mask_led|mask_pd)# kp kr l p 3
    mask_count = np.sum(mask_total,axis=(2,3)).reshape((kpos,krot,1,1))
    if weight_form =='mean':
        # weight = np.nansum(~mask_total[:,:,:,:,0],axis=(2,3)).reshape(kpos,krot,1,1) 
        weight = np.divide(mask_total,mask_count)
    elif weight_form =='weight':
        weight = np.ma.masked_array(light_f,mask_total[:,:,:,:,0])
        weight = np.power(weight,3)
        weight_sum = np.nansum(weight,axis=(2,3)).reshape((kpos,krot,1,1))
        weight = np.divide(weight,weight_sum)
    
    # 答案求平均（忽略nan）
    
    
    
    # weight = np.nansum(np.power(light_f,1/3),axis=(2,3)).reshape(kpos,krot,1,1) # kp kr
    # weight = np.divide(np.power(light_f,1/3),weight)
    # check = np.nansum(weight,axis=(2,3))
    # print(light_f[3,3,:,:])
    # print(check)
    
    # cross_led_av = np.multiply(cross_led,weight)
    # ori_sol_pd_coor = np.sum(np.multiply(cross_led,np.tile(weight,(3,1,1,1,1)).transpose((1,2,3,4,0))),axis=(2,3)).filled(fill_value=np.nan)
    # ori_sol_led_coor = np.sum(np.multiply(cross_pd,np.tile(weight,(3,1,1,1,1)).transpose((1,2,3,4,0))),axis=(2,3)).filled(fill_value=np.nan)
    ori_sol_pd_coor = np.nanmean(cross_led,axis = (2,3))#.filled(fill_value=np.nan) #kp kr 3,
    ori_sol_led_coor = np.nanmean(cross_pd,axis = (2,3))#.filled(fill_value=np.nan) #kp kr 3,
    # print(ori_sol_pd_coor.shape,ori_sol_led_coor.shape)
    
    
    # 由答案算in_ang,out_ang - ori_sol 3,  - ori_pd 3,pd
    sol_in_ang = np.arccos(np.inner(ori_sol_pd_coor,pd_ori_car.T)) # kp kr pd,
    sol_out_ang = np.arccos(np.inner(ori_sol_led_coor,led_ori_car.T)) #kp kr led,
    # print(sol_out_ang)
    sol_dis = np.sqrt(const * np.divide(np.multiply(\
                                                    np.tile(np.power(np.cos(sol_in_ang),pd_m),(led_num,1,1,1)).transpose(1,2,0,3),\
                                                    np.tile(np.power(np.cos(sol_out_ang),led_m),(pd_num,1,1,1)).transpose(1,2,3,0)\
                                                    ),\
                                        light_f)) #kp kr l p
        
    # check_dis = np.sqrt(np.sum(np.square(np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose(1,2,3,0,4)),axis=4))
    # check_dis = np.sum(~np.isclose(np.ma.masked_invalid(sol_dis),check_dis))
    # print('------------------------------------')
    # print('False dis:' ,check_dis)
    # print('------------------------------------')
    global sol_dis_av
    sol_dis_av= np.sum(np.multiply(sol_dis,weight),axis=(2,3))
    # print(sol_dis_av.shape,'~~~')
    sol_dis_av = np.nanmean(sol_dis,axis = (2,3))#kp kr
    # print()
    global error
    error = (np.sum(np.square(np.multiply(cross_led_av,sol_dis_av.reshape(kpos,-1,1))-glob_led_pos[:,:,0,:]),axis=2))
    global  unsolve
    global  solve
    solve = np.ma.count(error)
    unsolve = np.ma.count_masked(error)
    error = error.filled(np.inf) # masked改成inf
    error[error==0] = np.nan # sqrt不能處理0
    error = np.sqrt(error)
    error[np.isnan(error)]= 0
    

    
    global tolerance 
    global success
    success = np.sum(error<tolerance)
    global error_av 
    error_av = np.mean(error[error<tolerance])
    # print('unsolve:',unsolve,', solve:',solve,', success:',success)
    # print(np.average(sol_dis_av))
    
    # return glob_led_pos,glob_led_ori,error ,unsolve,success,error_av

# =============================================================================
# # 前置
# =============================================================================

# 從演算法裡面copy出來的有用資訊
sol_dis_av=[]
unsolve = 0
success = 0
error = []
error_av = []



# =============================================================================
# # set environment
# =======================================================================
threshold = 10**(-9)
tolerance = 0.1
effective = 80
weight_form = 'mean'

# 硬體參數
def set_hardware(led_hard,pd_hard):
    led_list = [\
                [1.7*np.pi]
                ]
    # respon area NEP darkcurrent shunt
    pd_list = [\
               [0.64, 6*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9],\
               [0.64, 5.7*10**(-6), 9*10**(-16), 5*10**(-12), 50*10**9],\
               [0.64, 33*10**(-6), 2*10**(-15), 50*10**(-12), 10*10**9],\
               [0.64, 100*10**(-6), 2.8*10**(-15), 200*10**(-12), 5*10**9],\
               [0.38, 36*10**(-6), 3.5*10**(-14), 100*10**(-12), 0.1*10**9]\
               ]
    pd_list = np.array(pd_list)
    print(pd_list[pd_hard,:])
    print(led_list[led_hard])
    return led_list[led_hard],pd_list[pd_hard,:]
led_hard = 0
pd_hard = 0
led_para,pd_para = set_hardware(led_hard, pd_hard)
led_pt = led_para[0]
pd_respon,pd_area,NEP,darkcurrent,shunt = pd_para
# led_pt = 1.7*np.pi #5A VSMA1085250
# pd_saturate = np.inf
# pd_area =1#*10**(-6) #BPW24R  #SFH203PFA
# pd_respon = 6*10**(-6)
# NEP = 10**(-14)
# background = 790*10**(-9)
# dark_current = 10**(-12)
# 演算法參數

bandwidth = 9.7**13/10 # 320-1000nm
# shunt = 1000*10**6 # 10-1000 mega

# =============================================================================
# # set system
# =============================================================================

pd_num = 8
led_num = 8
led_m = 1
pd_m = 1

# ans = np.zeros((14,14,5,5,5,5,2))
# numl = np.array([3,5,8,10,12,15])
# # numl = np.arange(3,16,1)
# nump = np.arange(8,9,1)
# ml = np.array([1,1.5,2,3,5])
# mp = np.array([1,1.5,2,3,5])
# alphal = np.deg2rad(np.arange(10,60,10))
# alphap = np.deg2rad(np.arange(10,60,10))

pd_alpha = np.deg2rad(10)#傾角
pd_beta = np.deg2rad(360/pd_num)#方位角
pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd

led_alpha = np.deg2rad(10)#傾角
led_beta = np.deg2rad(360/led_num)#方位角
led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
led_ori_car = ori_ang2cart(led_ori_ang) #3xled


# =============================================================================
# # set sample points
# =============================================================================
testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3,-1)) # 3x?
testp_rot = np.array([[np.pi,0,0],[0,np.pi,0]]).T
testp_rot  = np.deg2rad(np.mgrid[0:0:1j, 10:60:6j, 36:360:10j].reshape((3,-1)))
testp_rot = np.concatenate((testp_rot,np.array([[0,0,0]]).T ),axis=1)+np.array([[np.pi,0,0]]).T

kpos = testp_pos.shape[1]
krot = testp_rot.shape[1]

solve_mulmul()
count_kpos = np.nansum(error<tolerance,axis=1)/krot
count_krot = np.nansum(error<tolerance,axis=0)/kpos

fig = plt.figure(figsize=(12, 8))
colormap= plt.cm.get_cmap('YlOrRd')
normalize =  colors.Normalize(vmin=0, vmax=1)

ax = fig.add_subplot(1,3,1,projection='3d')
ax.set_box_aspect(aspect = (1,1,1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
ax.set_xlim3d(-1.5,1.5)
ax.set_ylim3d(-1.5,1.5)
ax.set_zlim3d(0,3)

sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap=colormap,norm = normalize)
ax.scatter(0,0,0,color='k',marker='x')

fig.colorbar(sc,shrink=0.3,pad=0.1)

ax = fig.add_subplot(1,3,3,projection='polar')

sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalize)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(sc, cax=cbar_ax)


ax.grid(True)

ax = fig.add_subplot(1,3,2,projection='3d')
ax.set_box_aspect(aspect = (1,1,1))
ax.grid(False)
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_axis_off()

# ax.scatter(0,0,0,color='k',marker='x')

u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,20))
x = 1*np.cos(u)*np.sin(v)
y = 1*np.sin(u)*np.sin(v)
z = 1*np.cos(v)
# sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

a = np.linspace(-1,1,21)
b = np.linspace(-1,1,21)
A,B = np.meshgrid(a,b)
c = np.zeros((21,21))
ax.plot_surface(A,B,c, color="grey",alpha=0.2)

a,b,c1 = ori_ang2cart(testp_rot[1:,:])

ax.quiver(0,0,0,0,0,-1,color='r')
ax.quiver(0,0,0,a,b,c1,color = 'b')

# ax = fig.add_subplot(1,3,2,projection='polar')
# # ax.axis('equal')



# sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')


# fig.colorbar(sc,shrink=0.3,pad=0.1)

# ax.grid(True)

# # fig.suptitle((f'L={led_num},P={pd_num},Ml={led_m},Mp={pd_m}'))







# ans = np.zeros((18,18,2))
# fig1 = plt.figure(figsize=(15, 15))
# fig2 = plt.figure(figsize=(15, 15))
# cm = plt.cm.get_cmap('rainbow')

# for numli in range(numl.size):
#     for numpi in range(nump.size):
        
#         # pd_alpha = alphap[alphapi]
#         # led_alpha = alphal[alphali]
#         # pd_num = nump[numpi]
#         # led_num = numl[numli]
#         led_num = numl[numli]
#         pd_num = nump[numpi]
        
#         pd_beta = np.deg2rad(360/pd_num)#方位角
#         pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
#         pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd

#         led_beta = np.deg2rad(360/led_num)#方位角
#         led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
#         led_ori_car = ori_ang2cart(led_ori_ang) #3xled
        
#         solve_mulmul()
#         count_kpos = np.sum((error<tolerance),axis=1)/testp_rot.shape[1]
#         count_krot = np.sum((error<tolerance),axis=0)/testp_pos.shape[1]
        
        
#         ans[numli,numpi,0] = success
#         ans[numli,numpi,1] = error_av
        
#         # cm = plt.cm.get_cmap('rainbow')

#         # ax = fig1.add_subplot(5,5,1+mli*5+mpi,projection='3d')
#         # ax.set_box_aspect(aspect = (1,1,1))
#         # ax.set_xlabel('x')
#         # ax.set_ylabel('y')
#         # ax.set_zlabel('z')
#         # ax.grid(True)
#         # ax.set_xlim3d(-1.5,1.5)
#         # ax.set_ylim3d(-1.5,1.5)
#         # ax.set_zlim3d(0,1.5)

#         # sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap='rainbow')
#         # ax.scatter(0,0,0,color='k',marker='x')

#         # fig1.colorbar(sc,shrink=0.3,pad=0.1)

#         # ax = fig2.add_subplot(5,5,1+mli*5+mpi,projection='polar')
#         # # ax.axis('equal')
        
#         # sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')
        
        
#         # fig2.colorbar(sc,shrink=0.3,pad=0.1)
        
#         # ax.grid(True)
        
#         # # ax = fig.add_subplot(1,3,2,projection='3d')
#         # # ax.set_box_aspect(aspect = (1,1,0.5))
#         # # ax.grid(False)
#         # # ax.set_xlim3d(-1,1)
#         # # ax.set_ylim3d(-1,1)
#         # # ax.set_zlim3d(0,1)
#         # # ax.xaxis.set_ticklabels([])
#         # # ax.yaxis.set_ticklabels([])
#         # # ax.zaxis.set_ticklabels([])
#         # # ax.set_axis_off()

#         # # ax.scatter(0,0,0,color='k',marker='x')

#         # # u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,20))
#         # # x = 1*np.cos(u)*np.sin(v)
#         # # y = 1*np.sin(u)*np.sin(v)
#         # # z = 1*np.cos(v)
#         # # # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
#         # # ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

#         # # a = np.linspace(-1,1,21)
#         # # b = np.linspace(-1,1,21)
#         # # A,B = np.meshgrid(a,b)
#         # # c = np.zeros((21,21))
#         # # ax.plot_surface(A,B,c, color="grey",alpha=0.2)

#         # # a,b,c1 = ori_ang2cart(testp_rot[1:,:])

#         # # ax.quiver(0,0,0,0,0,-1,color='r')
#         # # ax.quiver(0,0,0,a,b,c1,color = 'b')

#         # ax = fig.add_subplot(1,3,2,projection='polar')
#         # # ax.axis('equal')



#         # sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')


#         # fig.colorbar(sc,shrink=0.3,pad=0.1)

#         # ax.grid(True)

#         # fig.suptitle((f'L={led_num},P={pd_num},Ml={led_m},Mp={pd_m}'))

# c = 0 

# for numli in range(numl.size):
#     for numpi in range(nump.size):
#         for mli in range(ml.size):
#             for mpi in range(mp.size):
#                 for alphali in range(alphal.size):
#                     for alphapi in range(alphap.size):
#                         pd_alpha = alphap[alphapi]
#                         led_alpha = alphal[alphali]
#                         pd_num = nump[numpi]
#                         led_num = numl[numli]
#                         led_m = ml[mli]
#                         pd_m = mp[mpi]
                        
#                         pd_beta = np.deg2rad(360/pd_num)#方位角
#                         pd_ori_ang = np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
#                         pd_ori_car = ori_ang2cart(pd_ori_ang) #3xpd

#                         led_beta = np.deg2rad(360/led_num)#方位角
#                         led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
#                         led_ori_car = ori_ang2cart(led_ori_ang) #3xled
                        
#                         solve_mulmul()
#                         ans[numli,numpi,mli,mpi,alphali,alphapi,0] = success
#                         ans[numli,numpi,mli,mpi,alphali,alphapi,1] = error_av
#                         c = c+1
#                         print(c)







# =============================================================================
# 
# solve_mulmul()
# 
# 
# count_kpos = np.sum((error<tolerance),axis=1)/testp_rot.shape[1]
# count_krot = np.sum((error<tolerance),axis=0)/testp_pos.shape[1]
# 
# 
# 
# 
# 
# # plot sample points
# axis_color = 'lightgoldenrodyellow'
# 
# # colors = cm.rainbow()
# 
# fig = plt.figure(figsize=(12, 8))
# 
# cm = plt.cm.get_cmap('rainbow')
# 
# ax = fig.add_subplot(1,3,1,projection='3d')
# ax.set_box_aspect(aspect = (1,1,1))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.grid(True)
# ax.set_xlim3d(-1.5,1.5)
# ax.set_ylim3d(-1.5,1.5)
# ax.set_zlim3d(0,3)
# 
# sc = ax.scatter(testp_pos[0,:],testp_pos[1,:],testp_pos[2,:],c = count_kpos,cmap='rainbow',alpha=0.5)
# ax.scatter(0,0,0,color='k',marker='x')
# 
# fig.colorbar(sc,shrink=0.3,pad=0.1)
# 
# =============================================================================

# ax = fig.add_subplot(1,3,2,projection='3d')
# ax.set_box_aspect(aspect = (1,1,0.5))
# ax.grid(False)
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(0,1)
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# ax.set_axis_off()

# ax.scatter(0,0,0,color='k',marker='x')

# u, v = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,np.pi,20))
# x = 1*np.cos(u)*np.sin(v)
# y = 1*np.sin(u)*np.sin(v)
# z = 1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")

# a = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
# A,B = np.meshgrid(a,b)
# c = np.zeros((21,21))
# ax.plot_surface(A,B,c, color="grey",alpha=0.2)

# a,b,c1 = ori_ang2cart(testp_rot[1:,:])

# ax.quiver(0,0,0,0,0,-1,color='r')
# ax.quiver(0,0,0,a,b,c1,color = 'b')

# =============================================================================
# ax = fig.add_subplot(1,3,2,projection='polar')
# # ax.axis('equal')
# 
# 
# 
# sc = ax.scatter(testp_rot[2,:],np.rad2deg(testp_rot[1,:])  ,c = count_krot,cmap='rainbow')
# 
# 
# fig.colorbar(sc,shrink=0.3,pad=0.1)
# 
# ax.grid(True)
# 
# fig.suptitle((f'L={led_num},P={pd_num},Ml={led_m},Mp={pd_m}'))
# =============================================================================




# =============================================================================
# # =============================================================================
# # plot
# # =============================================================================
# 
# axis_color = 'lightgoldenrodyellow'
# 
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1,2,1,projection='3d')
# ax.set_box_aspect(aspect = (1,1,1))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.grid(True)
# ax.set_xlim3d(-1.5,1.5)
# ax.set_ylim3d(-1.5,1.5)
# ax.set_zlim3d(0,3)
# 
# 
# # Adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.25, bottom=0.25)
# 
# 
# 
# 
# # bubble = ax.scatter(testp_pos[:,:,0],testp_pos[:,:,1],testp_pos[:,:,2],size = error+0.1)
# # draw sphere
# # =============================================================================
# # u, v = np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,20))
# # x = 0.1*np.cos(u)*np.sin(v)
# # y = 0.1*np.sin(u)*np.sin(v)
# # z = 0.1*np.cos(v)
# # sphere = ax.plot_wireframe(x+testp_pos[0,:], y+testp_pos[1,:], z+testp_pos[2,:], color="w",alpha=0.2, edgecolor="#808080")
# # ax.plot_wireframe(x, y, z, color="w",alpha=0.2, edgecolor="#808080")
# # =============================================================================
# 
# arrow = 0.3*np.array([[0,0,1]]).T
# ax.quiver(0,0,0,arrow[0,:],arrow[1,:],arrow[2,:],arrow_length_ratio=0.5, color='k')
# arrow_rot = np.tile((testp_rot_matlist(testp_rot) @ arrow).squeeze(),(testp_pos.shape[1],1,1)) #kr 3 1
# arrow_p = np.tile(testp_pos,(testp_rot.shape[1],1,1)).transpose(2,0,1)
# # axis_item = ax.quiver(arrow_p[:,:,0],arrow_p[:,:,1],arrow_p[:,:,2],arrow_rot[:,:,0],arrow_rot[:,:,1],arrow_rot[:,:,2],arrow_length_ratio=0.5, color=["r"])
# 
# 
# glob_led_pos,glob_led_ori,error,unsolve,success,error_av  = solve_mulmul(led_num,pd_num,led_m,pd_m)
# text_item = ax.text(-2.5,-2.5,-2, f'Unsolvable:{unsolve}\nSuccess:{success}\nMean error:{error_av:.4E}')
# 
# 
# 
# error = error.flatten()
# bubble = []
# for i in range(error.size):
#     if error[i] != np.inf:
#         bubble.append(ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 10**9*error+10,c = 'b'))
#     else:
#         bubble.append(ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],marker = 'x',c = 'k',s = 100))
# 
# 
# # Add two sliders for tweaking the parameters
# text = ['led amount','pd amount','led m','pd m']
# init_val = np.array([led_num,pd_num,led_m,pd_m])
# min_val = [3,3,2,2]
# max_val = [20,20,70,70]
# sliders = []
# 
# 
# for i in np.arange(len(min_val)):
# 
#     axamp = plt.axes([0.84, 0.8-(i*0.05), 0.12, 0.02])
#     # Slider
#     # s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
#     
#     s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i],valstep=1)
#     # else:
#     #     s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
#     sliders.append(s)
# 
# 
# # Define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 
#     global  sphere,axis_item,ans
#     #ax.collections.remove(sphere)
#     led_num,pd_num,led_m,pd_m = sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val
#     
#     pd_alpha = np.deg2rad(35)#傾角
#     pd_beta = np.deg2rad(360/pd_num)#方位角
#     global pd_ori_ang 
#     pd_ori_ang= np.stack( (pd_alpha*np.ones(pd_num),(pd_beta*np.arange(1,pd_num+1))),0 )#2x?
#     global pd_ori_car 
#     pd_ori_car= ori_ang2cart(pd_ori_ang) #3xpd
# 
#     led_alpha = np.deg2rad(45)#傾角
#     led_beta = np.deg2rad(360/led_num)#方位角
#     global led_ori_ang 
#     led_ori_ang = np.stack( (led_alpha*np.ones(led_num),(led_beta*np.arange(1,led_num+1))),0 )#2x?
#     global led_ori_car 
#     led_ori_car= ori_ang2cart(led_ori_ang) #3xled
#     
#     glob_led_pos,glob_led_ori,error,unsolve,success,error_av = solve_mulmul(led_num,pd_num,led_m,pd_m )
# 
#     error = error.flatten()
#     for i in range(error.size):
#         ax.collections.remove(bubble[i])
#         text_item.set_text(f'Unsolvable:{unsolve}\nSuccess:{success}\nMean error:{error_av:.4E}')
#         if error[i] != np.inf:
#             bubble[i] = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],s = 10**9*error+10,c = 'b',)
#         else:
#             bubble[i] = ax.scatter(glob_led_pos[:,:,0,0],glob_led_pos[:,:,0,1],glob_led_pos[:,:,0,2],marker = 'x',c = 'k',s = 100)
# 
#     fig.canvas.draw_idle()
# 
# 
# for i in np.arange(len(min_val)):
#     #samp.on_changed(update_slider)
#     sliders[i].on_changed(sliders_on_changed)
# 
# 
# 
# plt.show()
# =============================================================================
