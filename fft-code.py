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

c = 3e17                # [nm/s] (if working with frequency this should be cahnged back to 3e8 m/s)

def DeltaPhiRetrievalProcedure(x, y, order = 2, keep_min_freq = 0.08, keep_max_freq = -1, side = "left", show_plots = False, fft_x_lim = [-1e-12, 1e-12]):
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
        plt.xlim(fft_x_lim)  # Limit the x-axis to the positive frequencies
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
        plt.tight_layout()
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
        plt.plot(x, np.polyval(coefficients, x), color='r', linestyle='--', label="Extracted phase")
        # plt.plot(x, -1.95698265e-05*x**3 +   6.79351537e-02*x**2 -7.89563528e+01*x +3.04750604e+04,label="Original simulated phase", color='orange', linestyle = '-.')
        
        plt.legend()
        # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
    return [x, coefficients]
    

# GIVEN x IS IN WAVELENGTHS
def ObtainBetaFromPhi(phi, length):
    return lambda var: phi(var) / length

def dOmega(beta, function_of):
    from sympy import diff
    import sympy as sp
    #print(beta)
    x = sp.symbols('x')
    expr = beta(x)
    #print("HERE")
    beta_1 = diff(expr, x)
    #print(beta_1) 
    return lambda vars: [beta_1.subs(x, var) for var in vars]
    
def V_g(beta):                                      # Input beta as a function of wavelength
    from sympy import diff
    import sympy as sp    
    x = sp.symbols('x')
    expr = beta(x)
    beta_1 = diff(expr, x)                          # Perform differentiation with respect to wavelength
    dLambda_dOmega = - x**2 / (2 * np.pi * c)
    beta_1 = beta_1 * dLambda_dOmega                # Multiply by the chain to make the diff. wrt omega
    return lambda vars: [1 / beta_1.subs(x, var) for var in vars]

def DBeta_dOmega(beta):
    from sympy import diff
    import sympy as sp 
    x = sp.symbols('x')
    expr = beta(x)
    beta_1 = diff(expr, x)                          # Perform differentiation with respect to wavelength
    dLambda_dOmega = - x**2 / (2 * np.pi * c)
    beta_1 = beta_1 * dLambda_dOmega                # Multiply by the chain to make the diff. wrt omega
    return lambda var: beta_1.subs(x, var)

def GVD(beta):                          # Input beta as a function on wavelength
    # Need to compute first and second derivative of beta wrt lambda:
    from sympy import diff
    import sympy as sp    
    x = sp.symbols('x')
    expr = beta(x)
    dBeta_dLambda = diff(expr, x)              # First derivative wrt lambda
    d2Beta_dLambda2 = diff(expr, x, 2)           # Second derivative wrt lambda

    # Also need first and second derivative of lambda wrt omega:
    dLambda_dOmega = - x**2 / (2 * np.pi * c)
    d2Lambda_dOmega2 = 2 * x**3 / (2 * np.pi * c)**2

    # The final expression d2Beta / dOmega2 is
    d2Beta_dOmega2 = d2Beta_dLambda2 * dLambda_dOmega**2 + dBeta_dLambda * d2Lambda_dOmega2
    return lambda vars: [d2Beta_dOmega2.subs(x, var) for var in vars]

def Big_D(beta): # Input beta as a function of wavelength
    from sympy import diff
    import sympy as sp 
    # Need to compute first and second derivative of beta wrt lambda:
    x = sp.symbols('x')
    expr = beta(x)
    beta_1 = diff(expr, x)              # First derivative wrt lambda
    beta_2 = diff(expr, x, 2)           # Second derivative wrt lambda

    # Need to compute dLambda_dOmega and d2Lambda_dLambdadOmega:
    dLambda_dOmega = - x**2 / (2 * np.pi * c)
    d2Lambda_dLambdadOmega = - x / (np.pi * c)

    D = d2Lambda_dLambdadOmega * beta_1 + dLambda_dOmega * beta_2
    return lambda vars: [D.subs(x, var) for var in vars]

def Obtain_n(beta):
    return lambda vars: [beta(var) * c / var for var in vars]


# I think all functions below here to *** are wrong
def dnOmega(beta, order=2):
    from sympy import diff
    import sympy as sp
    #print(beta)
    x = sp.symbols('x')
    expr = beta(x)
    #print("HERE")
    beta_1 = diff(expr, x, order=order)
    #print(beta_1) 
    return lambda vars: [beta_1.subs(x, var) for var in vars]

def Beta_2(beta):
    from sympy import diff
    import sympy as sp
    x = sp.symbols('x')
    expr = beta(x)
    #print("HERE")
    return lambda ls: [l**3 / (2 * np.pi**2 * (3e17)**2) * diff(expr, x, order=1).subs(x, l) + l**4/(2 * np.pi * 3e17)**2 * diff(expr, x, order=2).subs(x,l) for l in ls]

# ***

# Read in the data from a file. 
# data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/simulation-data-1.csv")
# x = data["wavelengths[nm]"]
# y = data["amplitude"]
# coefficients = DeltaPhiRetrievalProcedure(x, y, 3, keep_min_freq=0.001, fft_x_lim = [-0.5, 0.5])


data = pd.read_csv("/Users/jackmorse/Documents/University/Year 4/Semester 1/FYP/Physics-FYP/Sample-Data-1/b6-spectral-phase-in-wavelengths.csv")
x = data["wavelengths"] # data["angularFrequency[G rad/s]"] #
y = data["amplitude_lambda"] #data["amplitude"]

idx = np.array(np.where(x < 1350)).flatten()
x = x[idx]
y= y[idx]
# keep_min_frequence_omega = 0.05e-12
[x, coefficients] = DeltaPhiRetrievalProcedure(x, y, keep_min_freq=0.01, keep_max_freq=-1, side="right", order=3, fft_x_lim = [-0.5, 0.5])
import sympy as sp
phi = lambda var: np.poly1d(coefficients)(var)



l_b6 = 680*1e-3 # m
beta = ObtainBetaFromPhi(phi, l_b6)
v_g = V_g(beta)
gvd = GVD(beta)
D = Big_D(beta)

""" Plots """
plt.figure(figsize=(12, 8))
xlabel = "Wavelength, $\lambda$ [nm]"

# Plot 1: Phi
plt.subplot(2, 3, 1)
plt.plot(x, phi(x))
plt.title("Phi")
plt.xlabel(xlabel)

# Plot 2: Beta
plt.subplot(2, 3, 2)
plt.plot(x, beta(x), label="beta")
plt.title("Beta")
plt.xlabel(xlabel)

# Plot 3: n (Refractive index)
plt.subplot(2, 3, 3)
plt.plot(x, Obtain_n(beta)(x), label="n")
plt.legend()
plt.title("Refractive index")
plt.xlabel(xlabel)

# Plot 4: v_g (Group velocity)
plt.subplot(2, 3, 4)
plt.plot(x, v_g(x), label="v_g")
plt.title("Group velocity")
plt.xlabel(xlabel)

# Plot 5: GVD 
plt.subplot(2, 3, 5)
plt.plot(x, gvd(x), label="GVD")
plt.title("GVD")
plt.legend()
plt.xlabel(xlabel)

# Plot 6: D
plt.subplot(2, 3, 6)
plt.plot(x, D(x), label="D")
plt.title("D")
plt.legend()
plt.xlabel(xlabel)

# Show Plots
plt.tight_layout()
plt.show()

# plt.plot(x, Beta_2(beta)(x), label = "Beta2")
# plt.title("Beta_2 in wavelength")
# plt.legend()
# plt.show()
#n = beta/2*np.pi

