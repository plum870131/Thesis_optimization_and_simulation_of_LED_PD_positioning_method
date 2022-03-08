


import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

# 假設靜態，因此單點對環境

# # pd coor x1 不會動
# led 用scenario產生很多點
# test point個transfer matrix(homo)

# config for pd &led
# transfer led coor to pd coor
# estimate d,theta,psi
# calculate strendth (mxn)

# calculate pos
# calculate error
# objective

# update

#------easy-------
# 把一維向量變成3x1(垂直)的向量
def form(list1): #list = [a,b,c]
    list2 = np.array(list1)
    if len(list2.shape)==1:# 一維矩陣
        return np.transpose(np.array([list1]))
    elif len(list2.shape)==2: # ?x3的進來轉一圈  
        return np.transpose(np.array(list1))
def ori_ang2cart(ori_ang):#ori_ang = 2xsensor_num np.array, 第一列傾角 第二列方位
    return np.stack((\
    np.multiply(np.sin(ori_ang[0,:]), np.cos(ori_ang[1,:])),\
    np.multiply(np.sin(ori_ang[0,:]), np.sin(ori_ang[1,:])),\
    np.cos(ori_ang[0,:])    \
        ),0)

# 用法：rotate_x(pos,np.deg2rad(45))
def rotate_x(ang): #ang[rad](3*3)
    rot = np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])
    # print(rot)
    return rot #是一個matrix
def rotate_y(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad](3*3)
    rot = np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    # print(rot)
    return rot #是一個matrix
def rotate_z(ang): #mat[被旋轉的矩陣](3*n個點)，ang[rad](3*3)
    rot = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
    # print(rot)
    return rot #是一個matrix
def rotate_mat(ang_list):#ang_list[x,y,z]的角度in rad #順序是先轉x->y->z
    return np.dot( rotate_z(ang_list[2]),(np.dot(rotate_y(ang_list[1]),rotate_x(ang_list[0]))) ) 
def rotate(mat, ang_list):#mat 3x? #ang_list [a,b,c]
    return np.dot(rotate_mat(ang_list),mat)
def trans(mat,trans):
    if len(mat.shape)==2: #是二維矩陣
        (_,num)=mat.shape
        # print(mat)
        # print(np.transpose(np.tile(trans,(num,1))))
        return mat + np.tile(trans,(1,num))
    else: #bug
        print('translation error: 你可能忘了把轉移向量變成[[]]')
        return None

def homogeneous_mat(ang_list, trans): #trans(3,1)
    hom = np.zeros((4,4))
    hom[:3,:3]= rotate(ang_list)
    hom[3,3]=1
    hom[:3,3]=trans
    return hom

def inv_hom_mat(ang_list, trans):#trans(3,1)
    hom = np.zeros((4,4))
    hom[:3,:3]= np.transpose(rotate(ang_list))
    hom[3,3]=1
    hom[:3,3]=-1*np.dot(hom[:3,:3],trans)
    return hom

# =======================================================================
# Scenario

xx,yy,zz = np.array(np.meshgrid(np.arange(0, 3.3, 0.5), #2 x
                      np.arange(0, 3.3, 0.5), #1 y
                      np.arange(1, 3, 1))) #3 z

# [[x][y][z]] 3x?
testp_pos = np.stack((np.ndarray.flatten(xx),np.ndarray.flatten(yy),np.ndarray.flatten(zz)),0)                     
kpos = testp_pos[0].size # num of test position
print(kpos,'kpos')
# [[rotx][roty][rotz]] 3x?
testp_rot = np.stack((np.deg2rad(np.arange(180,270,15)),np.zeros(np.arange(180,270,15).size),np.zeros(np.arange(180,270,15).size)),0)
krot = testp_rot[0].size # num of test rotate orientation
print(krot,'krot')
# =======================================================================
# pd led: coor config
# wrt 自己的coordinate
pd_num = 5
pd_pos = (np.zeros((3,pd_num)))
alpha = np.deg2rad(20)#傾角
beta = np.deg2rad(360/pd_num)#方位角
pd_ori_ang = np.stack( (alpha*np.ones(5),(beta*np.arange(1,pd_num+1))),0 )
pd_ori_car = ori_ang2cart(pd_ori_ang)

#pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
pd_para = [2,1,1] #[0:M, 1:area, 2:respons]

led_num = 2
led_pos = form([[0,0,0],[1,0,0]])
led_ori_ang = np.array([np.deg2rad([0,30]),[0,0]])
led_ori_car = ori_ang2cart(pd_ori_ang)
led_para = [2,100]#led_para[0:m, 1:optical power]

# =======================================================================
# transfer led coor to pd coor

# 產生k2個rotation matrix
# shape(k2,3,3) 每個3x3是一個rotation matrix
def testp_rot_matlist(testp_rot): # testp_rot [[rotx][roty][rotz]] 3x?
    out = np.zeros((krot,3,3)) # shape(k2,3,3)
    for i in range(krot):
        out[i,:,:] = rotate_mat(testp_rot[:,i])
    return out ## shape(k2,3,3) 每個3x3是一個rotation matrix
# 把多個點轉到global coor上
# pos是多個點 3x?
# testp_rot是[[rotx][roty][rotz]] ，多個轉換參數
# return krotx3xm
def global_testp_after_rot(pos, testp_rot): #pos(or ori)[3x?] #testp_rot (krot,3,3)
    rot_list = testp_rot_matlist(testp_rot)
    # print(pos,'pos')
    # print(testp_rot,'prot')
    # print(testp_rot[0].size,'krot')
    # print(pos[0].size,'led)num 2')
    out = np.zeros((testp_rot[0].size,3,pos[0].size))
    for i in range(testp_rot[0].size):
        out[i,:,:] = np.dot(rot_list[i],pos)
    return out # krotx3xm

# out(krot,kpos,3,m) 
def global_testp_trans(pos , testp_pos): 
    # pos [krotx3xm] or [3xm]
    # testp_pos [3xkpos]
    if len(pos.shape)==3:
        kpos = testp_pos[0].size # kpos
        krot =  pos.shape[0] #krot
        m =  pos.shape[2] #led_num

        out = np.zeros((krot,kpos,3,m))
        # out = np.zeros((pos.shape[0],testp_pos[0].size,3,pos.shape[2]))
        for i in range(krot):
            out[i,:,:,:]= np.tile(pos[i,:,:],(kpos,1,1))+np.tile(testp_pos,(m,1,1)).transpose(2,1,0)
        return out
    elif len(pos.shape)==2:
        print('error in global_testp_trans')

#(krot,kpos,3,m) 
#先把led_num個pos位置經過krot次旋轉，變成krotx3xled_num的testpoints，再將所有testpoints平移到testp_pos上
glob_led_pos = global_testp_trans(global_testp_after_rot(led_pos,testp_rot), testp_pos)
# print(glob_led_pos,'me')
#(krot,3,led_num) 
glob_led_ori = global_testp_after_rot(ori_ang2cart(led_ori_ang),testp_rot)
# print(glob_led_ori.shape,'me')


# =======================================================================
# estimate d,theta,psi
def cal_d_in_out(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car,krot,kpos,led_num,pd_num):
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

        # krot kpos led pd  #glob_led_ori(krot,3,led_num) 
        # out_ang[:,:,:,pd] = -pos_delta[:,:,:,pd,0]* glob_led_ori[,0,] - pos_delta[:,:,:,pd,1]*   -pos_delta[:,:,:,pd,2]*
    for led in range(led_num):
        for r in range(krot):
            out_ang[r,:,led,:] = -pos_delta[r,:,led,:,0]*glob_led_ori[r,0,led] - pos_delta[r,:,led,:,1]*glob_led_ori[r,1,led] - pos_delta[r,:,led,:,2]*glob_led_ori[r,2,led]
    in_ang = np.arccos(np.divide(in_ang,dis))
    out_ang = np.arccos(np.divide(out_ang,dis))
    
    return dis,in_ang,out_ang

# krot x kpos x led x pd
dis,in_ang,out_ang = cal_d_in_out(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car,krot,kpos,led_num,pd_num)


'''check if d,in_ang,out_ang is right'''
'''
#krot2 kpos98 led2 pd5
a,b,c,d = 2,50,1,2

dis_cal,in_cal,out_cal = dis[a,b,c,d],in_ang[a,b,c,d],out_ang[a,b,c,d]

dis_real = np.sqrt(np.sum(np.square(glob_led_pos[a,b,:,c]-pd_pos[:,d])))
in_real = np.arccos(np.dot(glob_led_pos[a,b,:,c]-pd_pos[:,d],pd_ori_car[:,d])/dis_real)
out_real = np.arccos( np.dot( -glob_led_pos[a,b,:,c]+pd_pos[:,d],glob_led_ori[a,:,c] ) /dis_real )
print(dis_cal==dis_real)
print(in_cal==in_real)
print(out_cal==out_real)
# print(np.rad2deg(in_cal),np.rad2deg(in_real))
# print(np.rad2deg(out_cal),np.rad2deg(out_real))
'''


print('hi')

# =======================================================================
# calculate strendth 

# # dis,in_ang,out_ang   [krot x kpos x led x pd]


def cal_strength_current(dis,in_ang,out_ang,pd_para,led_para):
    # pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
    # dis,in_ang,out_ang   [krot x kpos x led x pd]
    # strength [krot x kpos x led x pd]
    # strength = np.zeros((krot,kpos,led_num,pd_num))
    led_m, area, respon = pd_para
    pd_m , power= led_para
    k = respon*power*(led_m+1)*area /(2*np.pi)
    return k*np.divide( np.multiply(\
                            np.power(np.cos(in_ang),pd_m),\
                            np.power(np.cos(out_ang),led_m)),\
                        np.square(dis))
    
print(cal_strength_current(dis,in_ang,out_ang,pd_para,led_para).shape)

# =======================================================================
# add noise

# =======================================================================
# calculate pos

# =======================================================================
# calculate error



# =======================================================================
# visualization
ax = fig = plt.figure().add_subplot(projection='3d')

# px,py,pz為三個list，分別紀錄所有要plot的位置分量
# pu, pv, pw為三個list，分別紀錄所有要plot的orientation分量

'''
px, py, pz = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))
pu = np.sin(np.pi * px) * np.cos(np.pi * py) * np.cos(np.pi * pz)
pv = -np.cos(np.pi * px) * np.sin(np.pi * py) * np.cos(np.pi * pz)
pw = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * px) * np.cos(np.pi * py) *
     np.sin(np.pi * pz))
'''
# Make the grid
px, py, pz = pd_pos
# Make the direction data for the arrows
pu,pv,pw = ori_ang2cart(pd_ori_ang)

ax.quiver(px, py, pz, pu, pv, pw, length=0.2, normalize=True,color='g')

# Make the grid
px, py, pz = glob_led_pos.transpose(2,0,1,3)
# Make the direction data for the arrows
pu,pv,pw = np.tile(glob_led_ori,(kpos,1,1,1)).transpose(2,1,0,3)

ax.quiver(px, py, pz, pu, pv, pw, length=0.2, normalize=True,color='r')

# setting
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 3])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("config")

# plt.show()

'''
fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')
for i in led_list:
    x,y,z = zip(i.point) #position
    u,v,w = zip(0.3*i.orien) #orientation
    ax.quiver(x,y,z,u,v,w,color='r')
    # ax.quiver(np.concatenate((i.point,i.point+i.orien),axis = 0),color = 'r')
for i in pd_list:
    x,y,z = zip(i.point)
    u,v,w = zip(0.3*i.orien)
    ax.quiver(x,y,z,u,v,w,color='g')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 0])
ax.set_title("config")

# plt.show()
'''








