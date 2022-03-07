import matplotlib
import numpy as np
from numpy.lib import meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sympy
min = 0
max = 10
step = 100
x = np.linspace(min,max,step)
y = np.linspace(min,max,step)
xx, yy = np.meshgrid(x,y)

stheta , sphi, sd = sympy.symbols('stheta,sphi,sd')
# k = (led_lamb+1)/(2*sympy.pi) *led_flux *pd_area
# Pr = sympy.Eq( k * sympy.Pow((sympy.cos(theta)),led_lamb) * sympy.Pow((sympy.cos(phi)),pd_lamb) / (sympy.Pow(d,2)),light[0])
Pr = sympy.Eq( stheta**2+sphi**2+sd**2,1)

ans = sympy.solve([Pr],(stheta,sphi,sd))

a = np.linspace(0,0.25*np.pi,100)
b = np.linspace(0,0.25*np.pi,100)
aa,bb = np.meshgrid(a,b)
c = np.divide( np.cos(aa),np.cos(bb))
print(c)

fig = plt.figure()
ax3d = plt.axes(projection="3d")

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(aa,bb,c, cmap='plasma')
plt.show()

# import numpy as np
# from matplotlib import pyplot as plt
# from scipy import signal


# # Filter requirements.
# T = 5.0         # Sample Period
# fs = 30.0       # sample rate, Hz
# cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
# nyq = 0.5 * fs  # Nyquist Frequency
# order = 2       # sin wave can be approx represented as quadratic
# n = int(T * fs) # total number of samples


# t = np.linspace(0,int(T),int(T*fs),endpoint=False)
# print(t)
# # sin wave
# sig = np.sin(1.2*2*np.pi*t)
# # Lets add some noise
# noise = 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
# data = sig + noise

# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients 
#     b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
#     y = signal.filtfilt(b, a, data)
#     return y

# # Filter the data, and plot both the original and filtered signals.
# y = butter_lowpass_filter(data, cutoff, fs, order)

# plt.plot(t,data)
# plt.plot(t,y)
# plt.show()











# SAMPLE_RATE = 44100  # Hertz
# DURATION = 10  # Seconds

# def generate_sine_wave(freq, sample_rate, duration):
#     x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
#     frequencies = x * freq
#     # 2pi because np.sin takes radians
#     y = np.sin((2 * np.pi) * frequencies)
#     return x, y

# # Generate a 2 hertz sine wave that lasts for 5 seconds
# x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

# t = np.linspace(0, DURATION, SAMPLE_RATE*DURATION, endpoint=False)
# square = (signal.square( 2 * np.pi * 0.5 * t)+1)*0.5
# _,noise = generate_sine_wave(100, SAMPLE_RATE, DURATION)
# noise = noise *0.2
# # sig = np.multiply(square,y)+noise
# sig = y + noise

# from scipy.signal import butter, lfilter
# order=5
# normalized_cutoff_freq=2 * 8 / SAMPLE_RATE
# numerator_coeffs, denominator_coeffs= signal.butter(order, normalized_cutoff_freq)
# filtered = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

# from scipy.fft import fft, fftfreq
# N = SAMPLE_RATE * DURATION

# yf = fft(sig)
# xf = fftfreq(N, 1 / SAMPLE_RATE)

# # plt.plot(xf, np.abs(yf))
# # plt.xlim(0,100)
# plt.plot(t,sig)
# plt.show()
