#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:58:49 2023

@author: jackmorse
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
import pandas as pd
from scipy import interpolate
from scipy.signal import hilbert
from DataAnalysisClass import DataHandling as DH

c = 3e17        # Speed of light in nm/s

data_set = "b6"
    
data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/" + data_set + "_fringes.csv", header=None, skiprows=40)
fringes_x = data[0]
fringes_y = data[1]

data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/" + data_set + "_ref.csv", header=None, skiprows=40)
ref_x = data[0]
ref_y = data[1]

data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/" + data_set + "_signal.csv", header=None, skiprows=40)
signal_x = data[0]
signal_y = data[1]

# convert from log and to frequenc
def ArrangeData(x, y): # x in nm
    y = 10**(y/10)
    plt.plot(x, y)
    plt.show()
    x2 = x
    x = 2 * np.pi * c / x           # Frequency in rad/s
    
    x_grid = np.linspace(min(x), max(x), len(y))  # Adjust the number of points as needed
    
    # Perform linear interpolation
    linear_interp = interpolate.interp1d(x, y, kind='linear')
    
    # Interpolate the data onto the grid
    y_interp = linear_interp(x_grid)
    return [x_grid, y_interp, x2, y]

[fringes_x, fringes_y, fringes_x_lambda, fringes_y_lambda] = ArrangeData(fringes_x, fringes_y)
[ref_x, ref_y, ref_x_lambda, ref_y_lambda] = ArrangeData(ref_x, ref_y)
[signal_x, signal_y, signal_x_lambda, signal_y_lambda] = ArrangeData(signal_x, signal_y)

plt.plot(fringes_x_lambda, (fringes_y_lambda))
plt.plot(ref_x_lambda,ref_y_lambda)
plt.plot(signal_x_lambda, signal_y_lambda)
plt.plot(signal_x_lambda, np.sqrt(signal_y_lambda * ref_y_lambda))
plt.title("b5 Spectrum Analyzer Traces")
plt.xlabel("Ang frequency, $\omega$")
plt.ylabel("Amplitude")
plt.show()
spectralPhase = ((fringes_y - ref_y - signal_y) + np.sqrt(ref_y * signal_y)) /(2 * np.sqrt(ref_y * signal_y))
spectralPhase_lambda = ((fringes_y_lambda - ref_y_lambda - signal_y_lambda) + np.sqrt(ref_y_lambda * signal_y_lambda)) /(2 * np.sqrt(ref_y_lambda * signal_y_lambda))
plt.plot(fringes_x_lambda, spectralPhase_lambda, label="Spectral phase")
plt.xlim([1050, 1350])
plt.ylim([-1.5, 1.5])
# plt.plot(fringes_x, np.sqrt(ref_y * signal_y), label="sqrt product")
plt.title("b5 after subtraction and division of ref and signal")
plt.xlabel("Ang frequency, $\omega$")
plt.legend()
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# envelope = hilbert(fringes_y)
# plt.plot(fringes_x, fringes_y)
# plt.plot(fringes_x, np.abs(envelope))
# plt.show()
file_path = "/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/" + data_set + "-spectral-phase-in-omega.csv"
DH.write_csv(file_path, [fringes_x_lambda, spectralPhase_lambda, fringes_x, spectralPhase], ["wavelengths[nm]", "amplitude_lambda", "angularFrequency[G rad/s]", "amplitude"], preamble = [])