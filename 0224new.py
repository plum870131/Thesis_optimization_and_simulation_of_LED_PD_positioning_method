# 從頭建兩個座標系之間的關係
import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

# self.pos = np.zeros(3)
# self.ori = np.array([0,0,-1])

# 把一維向量變成3x1(垂直)的向量
def form(list): #list = [a,b,c]
    return np.transpose(np.array([list]))

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
def rotate(ang_list):#ang_list[x,y,z]的角度in rad #順序是先轉x->y->z
    return np.dot( rotate_z(ang_list[2]),(np.dot(rotate_y(ang_list[1]),rotate_x(ang_list[0]))) ) 
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


class coordinate: #量測物與被量測物的座標系  物體移動時座標改變的是這裡
    def __init__(self):
        self.pos = np.zeros((3,1))
        self.ori = np.zeros((3,1))
    def __repr__(self): #print此object會顯示位置與指向
        return f"position:{self.pos}, orient:{self.ori}"
    def refresh_coor(self, pos, ori):
        self.pos = pos
        self.ori = ori

# 一個coordinate上有數個LED/PD
# 要有sensor list裝載sensor



class sensor(coordinate): #sensor的框架
    def __init__(self): #sensor_coor是選擇一個coordinate(pd/led)作為自己的coor
        self.pos = np.zeros((3,1))
        self.ori = np.zeros((3,1))
        self.num = 0
        self.par_list = [] #之後要加上參數config
        self.sensor_list =[] #補上位置與角度
    def config(self):
        pass
    def global_coor():
        pass
    # init後已經有此sensor相對coor的位置了，也就是組態，不會隨時間改變
    # 計算彼此關係、硬體參數建立
    # 1. 把self.pos&ori轉成pd座標系


pd_coor = sensor()
led_coor = sensor()
print(pd_coor.pos)
print(pd_coor.ori)

# pd coor x1 不會動
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
# pd led: coor config
# transfer led coor to pd coor
# estimate d,theta,psi
# calculate strendth (mxn)

# calculate pos
# calculate error
