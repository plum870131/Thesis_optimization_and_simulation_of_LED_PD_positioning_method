
from turtle import back
from funcfile import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.cm as cm
import matplotlib.colors as colors
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1,1,1)

# to = [0.01, 0.05, 0.2,0.035,0.0230,\
#     0.0643,0.0843,0.0932,0.0733,0.0435\
#     ,0.388,0.5,0.3678,0.2879,0.2086]
# to = np.sort(np.array(to))
# to_nt = [11776,41498,49383,2893,27164,\
#     47479,48511,48797,48063,42695,50256\
#     ,50320, 50231 ,50140,49925 ]
# to_nt = np.sort(np.array(to_nt))
# ax.set_xlabel('容許範圍(m)')

# to = [1000,370*10**3,10**6,10**9,1.6156*10**6\
#     ,1.487*10**7,1.078*10**8,1.08*10**5,9.88*10**3]
# to_nt = [49389,41496,36653,450,37792\
#     ,15729,3356,47504,49163]
# to = np.sort(np.array(to))[::-1]
# to_nt = np.sort(np.array(to_nt))
# ax.set_xlabel('頻寬(Hz)')
# ax.set_xscale('log')

to = [2.471,2.714,2.297,\
    2.037,1.782,1.493,\
        1.14,1,1.395,\
    1.32,1.262,1.198,1.065,1.094]
to_nt = [442,434,846,\
    1160,1951,5580,\
        47079,49261,8811,\
    13647,20707,32154,49139,48933]
ax.scatter(to,to_nt)
to = np.sort(np.array(to))[::-1]
to_nt = np.sort(np.array(to_nt))
ax.set_xlabel('多重路徑增益')

ax.set_ylabel('於容許範圍內的樣本點數量')
ax.plot(to,to_nt)


plt.show()