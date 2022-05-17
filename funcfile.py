import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

# -------------------------- function def ------------------------------------

'''形式轉換'''
# 把一維向量變成3x1(垂直)的向量
def form(list1): #list = [a,b,c]
    list2 = np.array(list1)
    if len(list2.shape)==1:# 一維矩陣
        return np.transpose(np.array([list1]))
    elif len(list2.shape)==2: # ?x3的進來轉一圈  
        return np.transpose(np.array(list1))

'''座標轉換 matrix '''
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

'''coordinate transfer'''
# 產生k2個rotation matrix
# shape(k2,3,3) 每個3x3是一個rotation matrix
def testp_rot_matlist(testp_rot): # testp_rot [[rotx][roty][rotz]] 3x?
    krot = testp_rot.shape[1]
    out = np.zeros((krot,3,3)) # shape(k2,3,3)
    for i in range(krot):
        out[i,:,:] = rotate_mat(testp_rot[:,i])
    return out ## shape(k2,3,3) 每個3x3是一個rotation matrix

def testp_euler_matlist(testp_rot): # testp_rot [[roll][pitch][yaw]] 3x?
    krot = testp_rot.shape[1]
    out = np.zeros((krot,3,3)) # shape(k2,3,3)
    for i in range(krot):
        out[i,:,:] = rotate_mat(testp_rot[:,i])
    return out ## shape(k2,3,3) 每個3x3是一個rotation matrix

# 把多個點轉到global coor上
# pos是多個點 3x?
# testp_rot是[[rotx][roty][rotz]] ，多個轉換參數
# return krotx3xm
def global_testp_after_rot(pos, testp_rot): #pos(or ori)[3x?] #testp_rot (krot,3)
    rot_list = testp_rot_matlist(testp_rot) # (krot,3,3)
    out = rot_list @ pos
    return out # krotx3xm

# out(krot,kpos,3,m) 
def global_testp_trans(pos , testp_pos): 
    # pos [krotx3xm] or [3xm]
    # testp_pos [3xkpos]
    if len(pos.shape)==3:
        kpos = testp_pos[1].size # kpos
        krot =  pos.shape[0] #krot
        m =  pos.shape[2] #led_num

        out = np.tile(pos,(kpos,1,1,1)).transpose((0,1,3,2))+\
              np.tile(testp_pos.T,(krot,pos.shape[2],1,1)).transpose((2,0,1,3))
        return out # kpos krot m 3
    elif len(pos.shape)==2:
        print('error in global_testp_trans')


'''計算d,in_ang,out_ang'''
def cal_d_in_out(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car,krot,kpos,led_num,pd_num):
    #glob_led_pos [krot kpos 3 led_num]
    #glob_led_ori [krot kpos 3 led_num]
    #pd_pos [3 pd_num]
    #pd_ori_car [3 pd_num]
    #dis = np.zeros((krot,kpos,led_num,pd_num))
    in_ang = np.zeros((krot,kpos,led_num,pd_num))
    out_ang = np.zeros((krot,kpos,led_num,pd_num))

    #pos_delta = np.zeros((krot,kpos,led_num,pd_num,3)) #led-pd: pd pointint to led
    pos_delta = np.tile(glob_led_pos,(pd_num,1,1,1,1)).transpose((1,2,4,0,3)) \
        - np.tile(pd_pos,(krot,kpos,led_num,1,1)).transpose((0,1,2,4,3))
    dis = np.sqrt(np.sum(np.square(pos_delta),axis=4)) # krot,kpos,led_num,pd_num
    in_ang = np.inner(pos_delta,pd_ori_car.T)
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
'''
def cal_d_in_out(glob_led_pos,glob_led_ori,pd_pos,pd_ori_car,krot,kpos,led_num,pd_num):
    #glob_led_pos [krot kpos 3 led_num]
    #glob_led_ori [krot kpos 3 led_num]
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
'''
'''計算strength'''
def cal_strength_current(dis,in_ang,out_ang,led_para,pd_para):
    # pd_para[0:M, 1:area, 2:respons] led_para[0:m, 1:optical power]
    # dis,in_ang,out_ang   [krot x kpos x led x pd]
    # strength [krot x kpos x led x pd]
    # strength = np.zeros((krot,kpos,led_num,pd_num))
    pd_m, area, respon = pd_para
    led_m , power= led_para
    k = respon*power*(led_m+1)*area /(2*np.pi)
    
    return k*np.divide( np.multiply(\
                            np.power(   np.multiply(np.cos(in_ang)>0,np.cos(in_ang))    ,pd_m),\
                            np.power(   np.multiply(np.cos(out_ang)>0,np.cos(out_ang))   ,led_m)),\
                        np.square(dis))

'''add noise'''
#return一個跟strength大小一樣的bias vector
def bias(strength,bias_val): # strength[(krot, kpos, led_num, pd_num)] bias:int
    return bias_val*np.ones(strength.shape)





def lamb_order(semi): #semi power angle in degree
    return -1*np.log(2)/np.log(np.cos(np.deg2rad(semi)))