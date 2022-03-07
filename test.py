import numpy as np
import matplotlib.pyplot as plt
import funcfile as func
import sympy
from scipy.optimize import fsolve,root
import math
sympy.init_printing()

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

a = np.transpose(np.array([[2,3,1]]))
b = np.vstack([a,[4]])
print(b)
print(b[:3,:])

print(np.dot(homogeneous_mat([0,0,-0.5*np.pi],np.array([[-4,6,0]])),np.transpose(np.array([[2,0,1,1]]))))