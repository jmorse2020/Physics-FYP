#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:41:53 2023

@author: jackmorse
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/simulation-data-1.csv")
wavelengths = data["wavelengths[nm]"]
amplitude = data["amplitude"]


# Sample data and parameters
N = len(wavelengths)  # Number of data points
T = (max(wavelengths) - min(wavelengths)) / N  # Sample spacing
x = wavelengths
y = amplitude

# Perform the FFT
yf = np.fft.fft(y)
xf = np.fft.fftfreq(N, T)
xf = np.fft.fftshift(xf)
yplot = np.fft.fftshift(yf)



# Plot the FFT results
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Signal")
plt.subplot(2, 1, 2)
plt.plot(xf, yplot) # Normalised
plt.title("FFT")
plt.xlim(-1, 1)  # Limit the x-axis to the positive frequencies
plt.xlabel("Frequency (Hz)")
plt.tight_layout()

# Define the frequency range to keep (e.g., 0.05 to 0.15 Hz)
keep_min_freq = 0.01
keep_max_freq = 0.5

# Filter the FFT to keep only the desired frequencies
filtered_yf = yplot

idx = np.where((keep_min_freq < xf) & (xf < keep_max_freq))
# print(idx)

# print(filtered_x)
filtered_yf = yplot[idx]
wanted_ys = np.zeros(len(yplot))
for i in range(0, len(yplot)):
    if np.isin(i, idx):
        wanted_ys[i] = 1
    else:
        wanted_ys[i] = 0
    
print(wanted_ys)
print("...")
print(len(wanted_ys))
# filtered_yf[(xf < -keep_max_freq) | (xf > -keep_min_freq) | (xf < keep_min_freq) | (xf > keep_max_freq)] = 0
wanted_frequencies = np.zeros(len(wavelengths))
wanted_frequencies[idx] = 1

plt.subplot(2, 1, 2)
plt.plot(xf[idx], yplot[idx], color='r')
plt.show()


# Perform the inverse FFT
filtered_y = np.fft.ifft(np.fft.ifftshift(filtered_yf))
count = 0
for i in range(0, len(wanted_ys)):
    if wanted_ys[i] != 0:
        wanted_ys[i] = filtered_y[count]
        count = count + 1
#  Pad on min(idx) zeros before and N - max(idx) zeros after

# N = len(filtered_x)  # Number of data points
# T = (max(filtered_x) - min(filtered_x)) / N
# x_back = np.fft.fftfreq(N, T)
# x_back = np.fft.fftshift(x_back)


# Plot the original data and the filtered data in the original domain
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Original Signal")
plt.subplot(2, 1, 2)
plt.plot(x, np.abs(wanted_ys))
plt.xlim([1040, 1042])
plt.title("Filtered Signal")
plt.tight_layout()
plt.show()