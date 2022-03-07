
import numpy as np
import matplotlib as plt

# 創造一個z為常數的平面，xy都是從min到max
def createPoints_plane0(min,max,step=1000,z=0):
    x = np.linspace(min,max,step)
    y = np.linspace(min,max,step)
    xx, yy = np.meshgrid(x,y)
    z = np.zeros((step,step))
    nx = np.zeros((step,step))
    ny = np.zeros((step,step))
    nz = np.zeros((step,step))
    return xx , yy, z, nx, ny, nz




# 已完成：把台北的airmass隨日期點完
def plot_airmass_taipei(): #lag緯度
    lag = 25
    ang = -23.5*np.sin(2*np.pi* np.arange(365) /365)
    am = 1/ np.cos(np.deg2rad(ang+lag))
    
    plt.plot(np.arange(am.size)+1,am)
    plt.title('台北Air Mass')
    plt.xlabel('日')
    plt.ylabel('Air Mass')
    plt.show()

def lamb_order(semi):#semi power angle in degree
    return -1*np.log(2)/np.log(np.cos(np.deg2rad(semi)))

def plot_lamb_effect():
    plt.figure()
    semi=np.linspace(10,60,51) #10~60的semi angle
    lamb = lamb_order(semi) #10~60度semi對應的lamb
    plt.subplot(2,1,1)
    plt.plot(semi,lamb)
    plt.title('Lambertian Order')
    plt.xlabel('Semi-power angle(deg)')
    plt.ylabel('Lambertian order m')

    index = semi[0::5]#5,10,15....的semi angle
    m =lamb[0::5]#5,10,15....semi對應的m
    plt.scatter(index,m)
    for i in range(len(m)):
        plt.annotate('m='+str(round(m[i],2)),(index[i],m[i]))
    plt.subplot(2,1,2)
    
    for i in range(10):
        degree = np.linspace(-90,90,200)
        lab='m:'+str(round(m[i],2))+', semi:'+str(index[i])
        curve = (m[i]+1)/(2*np.pi)*np.power(np.cos(np.deg2rad(degree)),m[i])
        plt.plot(degree , curve ,label = lab)
    plt.legend()
    plt.title('Relative Signal Strength')
    plt.xlabel('感測器所在角度(deg)')
    plt.ylabel('Relative Signal Strength')
    plt.tight_layout()
    plt.show()