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

c = 3e17                            # Speed of light in nm/s

data_set = "b6"
if ~isinstance(data_set, str):
    TypeError("'data_set' is not a string") 
x_axis_variable = "frequency"      # Options: "wavelength" or "frequency" or "omega"
custom_xlim = None # [1200, 1500]
custom_ylim = [-1.5, 1.5]

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
def ExtractUsefulData(x, y, x_axis_variable, show_plots = False): # x in nm
    y = 10**(y/10)
    if x_axis_variable.lower() == "wavelengths":
        x_grid = x
        y_interp = y
    elif x_axis_variable.lower() == "frequency" or x_axis_variable.lower() == "omega":
        x = 2 * np.pi * c / x                                           # Frequency in rad/s
        x_grid = np.linspace(min(x), max(x), len(y))                    # Adjust the number of points as needed        
        linear_interp = interpolate.interp1d(x, y, kind='linear')       # Perform linear interpolation
        y_interp = linear_interp(x_grid)                                # Interpolate the data onto the grid
    else: 
        print("Error: Check the spelling of input argument 'x_axis_variable'.")
        return [None, None]
    if show_plots:
        plt.plot(x_grid, y_interp)
        plt.title("Extracted Data")
        plt.xlabel(x_axis_variable)
        plt.show()
    return [x_grid, y_interp]

[fringes_x, fringes_y] = ExtractUsefulData(fringes_x, fringes_y, x_axis_variable)
[ref_x, ref_y] = ExtractUsefulData(ref_x, ref_y, x_axis_variable)
[signal_x, signal_y] = ExtractUsefulData(signal_x, signal_y, x_axis_variable)

plt.plot(fringes_x, fringes_y)
plt.plot(ref_x, ref_y)
plt.plot(signal_x, signal_y)
plt.plot(signal_x, np.sqrt(signal_y * ref_y))
plt.title(data_set + " Spectrum Analyzer Traces")
if x_axis_variable.lower() == "wavelengths":
    plt.xlabel("Wavelengths. $\lambda$ [nm]")
elif x_axis_variable.lower() == "frequency" or x_axis_variable == "omega":
    plt.xlabel("Angular frequency, $\omega$ [rad s$^-1$]")
else:
    print("'x_axis_variable' is not one of the expected values.")
plt.ylabel("Amplitude")
plt.show()
spectralPhase = ((fringes_y - ref_y - signal_y) + np.sqrt(ref_y * signal_y)) /(2 * np.sqrt(ref_y * signal_y))
# spectralPhase_lambda = ((fringes_y_lambda - ref_y_lambda - signal_y_lambda) + np.sqrt(ref_y_lambda * signal_y_lambda)) /(2 * np.sqrt(ref_y_lambda * signal_y_lambda))
plt.plot(fringes_x, spectralPhase, label="Spectral phase")
if custom_xlim is not None:
    try:        
        plt.xlim(custom_xlim)
    except:
        print("'custom_xlim' does not take correct values.")
if custom_ylim is not None:
    try:        
        plt.ylim(custom_ylim)
    except:
        print("'custom_ylim' does not take correct values.")

# plt.plot(fringes_x, np.sqrt(ref_y * signal_y), label="sqrt product")
plt.title(data_set + "after subtraction and division of ref and signal")
plt.xlabel("Ang frequency, $\omega$")
plt.legend()
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# envelope = hilbert(fringes_y)
# plt.plot(fringes_x, fringes_y)
# plt.plot(fringes_x, np.abs(envelope))
# plt.show()
file_path = "/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/" + data_set + "-spectral-phase-in-" + x_axis_variable + ".csv"
DH.write_csv(file_path, [fringes_x, spectralPhase], [x_axis_variable, "amplitude"], preamble = [])