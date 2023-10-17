#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:41:53 2023

@author: jackmorse
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# Read in the data from a file.
data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/simulation-data-1.csv")
x = data["wavelengths[nm]"]
y = data["amplitude"]


# Construct the Fourier Domain
N = len(x)                      # Number of data points
T = (max(x) - min(x)) / N       # Sample spacing
xf = np.fft.fftfreq(N, T)       # Create the Fourier domain
xf = np.fft.fftshift(xf)        # Shift the domain to be centered around 0

# Perform the FFT
yf = np.fft.fft(y)
yf = np.fft.fftshift(yf)

# Select range in Fourier domain to transform back
keep_min_freq = 0.01
keep_max_freq = 0.5

# Filter the FFT to keep only the desired frequencies
idx = np.where((keep_min_freq < xf) & (xf < keep_max_freq))
filtered_yf = yf[idx]
wanted_ys = np.zeros(len(yf))
wanted_ys[idx] = 1

# Perform the inverse FFT
filtered_y = np.fft.ifft(np.fft.ifftshift(filtered_yf))
count = 0
for i in range(0, len(wanted_ys)): # Merge the wanted values with an array of length same as original
    if wanted_ys[i] != 0:
        wanted_ys[i] = filtered_y[count]
        count += 1

# Plot the FFT results
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Signal")
plt.subplot(2, 1, 2)
plt.plot(xf, yf, label = "Full fft") # Normalised
plt.title("FFT")
plt.xlim(-1, 1)  # Limit the x-axis to the positive frequencies
plt.xlabel("Fourier Domain")
plt.tight_layout()
plt.subplot(2, 1, 2)
plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
plt.legend()
plt.show()


# Plot the original data and the filtered data in the original domain
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Original Signal")
plt.subplot(2, 1, 2)
plt.plot(x, np.abs(wanted_ys)**2)
plt.xlim([x[np.nonzero(wanted_ys)[0][0]], x[np.nonzero(wanted_ys)[0][-1]]])
plt.title("Filtered Signal (ifft of selected region)")
plt.tight_layout()
plt.show()
