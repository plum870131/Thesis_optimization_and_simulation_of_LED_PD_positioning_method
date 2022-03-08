import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

krot, kpos,led_num,pd_num=2,4,5,6

glob_led_pos = np.zeros((2,4,3,5)) #krot kpos 3 led
pd_pos = np.tile(100*np.arange(0,6),(3,1)) #3 pd
pd_ori_car = np.tile(100*np.arange(0,6),(3,1)) #3 pd
led_ori_car = np.tile(100*np.arange(0,5),(3,1)) #3 pd
out = np.zeros((2,4,5,6))#(krot,kpos,led_num,pd_num)

for i in range(2):
    for j in range(4):
        glob_led_pos[i,j,:,:] = glob_led_pos[i,j,:,:]+(4*i+j)

dis = np.zeros((krot,kpos,led_num,pd_num))
in_ang = np.zeros((krot,kpos,led_num,pd_num))
out_ang = np.zeros((krot,kpos,led_num,pd_num))

pos_delta = np.zeros((krot,kpos,led_num,pd_num,3)) #led-pd: pd pointint to led

for pd in range(pd_num):
    for led in range(led_num):
        # (x-x)^2 sqrt
        # glob_led_pos[:,:,:,led]  krotxkposx3
        pd_extend = np.tile(pd_pos,(krot,kpos,1,1))
        pos_delta[:,:,led,pd,:] = glob_led_pos[:,:,:,led]-pd_extend[:,:,:,pd]#krotxkposx3
        dis[:,:,led,pd] = np.sqrt(np.square(pos_delta[:,:,led,pd,:]).sum(axis=2))
        # in_ang[:,:,led,pd]= 

for pd in range(pd_num):
    # 計算該pd與all testpoints的角度
    in_ang[:,:,:,pd] = pos_delta[:,:,:,pd,0]*pd_ori_car[0,pd]+pos_delta[:,:,:,pd,1]*pd_ori_car[1,pd]+pos_delta[:,:,:,pd,2]*pd_ori_car[2,pd]
for led in range(led_num):
    out_ang[:,:,led,:] = -pos_delta[:,:,led,:,0]*led_ori_car[0,led] - pos_delta[:,:,led,:,1]*led_ori_car[1,led] - pos_delta[:,:,led,:,2]*led_ori_car[2,led]
in_ang = np.divide(in_ang,dis)
out_ang = np.divide(out_ang,dis)
print(in_ang.shape)
# print(a)
# print(a[:,:,:,0])

# print(np.sqrt(np.multiply(a,a)))


