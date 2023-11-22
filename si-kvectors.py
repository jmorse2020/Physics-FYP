#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:16:57 2023

@author: jackmorse
"""

import numpy as np
import matplotlib.pyplot as plt
from RefractiveIndexClass import RefractiveIndex as RI
from DataAnalysisClass import DataHandling as DH

def m2nm(x):
    return x * 1e9

# Parameters
wavelengths = np.linspace(950, 1130, 10000) # [nm]
ZDW = 1280                          # Zero delay wavelngth [nm]
delta = ZDW / 4                     # phase shift
central_wavelength = 1040           # [nm]
sigma = 20                          # Bandwidth
L_f = 0.2                           # Fibre length [m]
L_f = m2nm(L_f)                     # Fibre length [nm]
L_air = - L_f * RI.n_group(RI.n_fs, ZDW) + delta                               # Required air difference for ZDW

# Interfernce
deltaPhi = 2 * np.pi / wavelengths * (L_air  + L_f * RI.n_fs(wavelengths))     # Spectral phase difference
Gaussian = np.exp(- ((wavelengths - central_wavelength)/ sigma)**2)            # Modelled as a Gaussian spectrum

# Plots
plt.plot(wavelengths, deltaPhi) 
coefficients = np.polyfit(wavelengths, deltaPhi, 3)
plt.plot(wavelengths, np.polyval(coefficients, wavelengths), color='r', linestyle='--', label="Fit")
print(coefficients)
plt.show()
# plt.plot(wavelengths, np.cos(deltaPhi / 2)**2)

plt.plot(wavelengths, Gaussian * np.cos(deltaPhi / 2)**2)
plt.show()

file_path = "/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/simulation-data-1.csv"
DH.write_csv(file_path, [wavelengths, np.cos(deltaPhi /2)**2], ["wavelengths[nm]", "amplitude"], preamble = [])
# D2 = (2 * np.pi / ZDW**2) * (2 * L_air / ZDW + 2 * L_f * (RI.n_fs(ZDW) / ZDW - RI._deriv(RI.n_fs, ZDW)))
# D1 = - 2 * np.pi / (ZDW**2) * (L_air + RI.n_group(RI.n_fs, ZDW) * L_f)
# print("D1 = ", D1 *1e9)