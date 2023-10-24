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
import cmath

def DeltaPhiRetrievalProcedure(x, y, order = 2, keep_min_freq = 0.08, keep_max_freq = -1, side = "left", show_plots = True):
    # Construct the Fourier Domain
    N = len(x)                      # Number of data points
    T = (max(x) - min(x)) / N       # Sample spacing
    xf = np.fft.fftfreq(N, T)       # Create the Fourier domain
    xf = np.fft.fftshift(xf)        # Shift the domain to be centered around 0

    # Perform the FFT
    yf = np.fft.fft(y)
    yf = np.fft.fftshift(yf)

    # Filter the FFT to keep only the desired frequencies
    if side == "both":
        if keep_max_freq == -1: # Go to max
            idx_left = np.array(np.where(xf < -keep_min_freq)).flatten()                                # left of DC
            idx_right = np.array(np.where(keep_min_freq < xf)).flatten()                                # right of DC
        else:
            idx_left = np.array(np.where((-keep_max_freq < xf) & (xf < -keep_min_freq))).flatten()      # left of DC
            idx_right = np.array(np.where((keep_min_freq < xf) & (xf < keep_max_freq))).flatten()       # right of DC
        idx = (np.concatenate((idx_left, idx_right)))
    elif side == "right":
        if keep_max_freq == -1: # Go to max
            idx = np.array(np.where(keep_min_freq < xf)).flatten()                                # right of DC
        else:
            idx = np.array(np.where((keep_min_freq < xf) & (xf < keep_max_freq))).flatten() 
    elif side == "left":
        if keep_max_freq == -1: # Go to max
            idx = np.array(np.where(xf < -keep_min_freq)).flatten()                                # left of DC                           # right of DC
        else:
            idx = np.array(np.where((-keep_max_freq < xf) & (xf < -keep_min_freq))).flatten()      # left of DC
    else:
        print("'side' is not a valid argument. It should be 'left', 'right', or 'both'.")
        return
    
    # Define the box filter
    box_filter = np.zeros(len(yf), dtype=complex)
    box_filter[idx] = 1
    filtered_fourier_data = yf * box_filter

    # Perform the inverse FFT
    filtered_y = np.fft.ifft(np.fft.ifftshift(filtered_fourier_data))

    # Plot the FFT results
    if show_plots == True:
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x, y)
        plt.title("Signal")
        plt.subplot(2, 1, 2)
        plt.plot(xf, yf, label = "Full fft") # Normalised
        plt.title("FFT")
        plt.xlim(-1e-12, 1e-12)  # Limit the x-axis to the positive frequencies
        plt.xlabel("Fourier Domain")
        plt.tight_layout()
        plt.subplot(2, 1, 2)
        if side == "both":
           plt.plot(xf[idx_left], yf[idx_left], color='r', label = "Selected region")
           plt.plot(xf[idx_right], yf[idx_right], color='r')   
        elif side == "right":
            plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
        elif side == "left":
            plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
        plt.legend()
        plt.show()


        # Plot the original data and the filtered data in the original domain
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x, y)
        plt.title("Original Signal")
        plt.subplot(2, 1, 2)
        plt.plot(x, np.abs(filtered_y)**2, label="filtered_y")
        # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
        plt.title("Filtered Signal (ifft of selected region)")
        plt.tight_layout()
        plt.legend()
        plt.show()

    # Extract phase and unwrap
    final_ys = np.zeros(len(filtered_y))
    for i in range(len(filtered_y)):
        final_ys[i] = cmath.phase((filtered_y[i]))
    final_ys = np.unwrap(final_ys)
    
    # Perform the fit
    coefficients = np.polyfit(x, final_ys, order)

    if show_plots == True:
        plt.plot(x, final_ys)
        plt.title("$\Phi(\omega)$")
        plt.xlabel("$\omega$")
        plt.ylabel("Intensity")
        plt.plot(x, np.polyval(coefficients, x), color='r', linestyle='--', label="Fit")
        plt.legend()
        # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
    return coefficients
    

# # Read in the data from a file. 
# data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/simulation-data-1.csv")
# x = data["wavelengths[nm]"]
# y = data["amplitude"]
# coefficients = DeltaPhiRetrievalProcedure(x, y)


data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/b5-spectral-phase-in-omega.csv")
x = data["angularFrequency[G rad/s]"] # data["wavelengths[nm]"]
y = data["amplitude"] #data["amplitude"]
idx = np.array(np.where(x > 1.5e15)).flatten()
x = x[idx]
y= y[idx]
coefficients = DeltaPhiRetrievalProcedure(x, y, keep_min_freq=0.01e-12, keep_max_freq=-1, side="right", order=3)