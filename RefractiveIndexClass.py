#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:43:25 2023

@author: jackmorse


Refractive indices of known materials.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class RefractiveIndex:
    def __init__():
        pass
    def _nm2um(nm):
        return nm / 1000
    
    def n_fs(variable, parameter = "wavelength"):
        """
        refractive index of fused silica.
        
        Fused Silica based on Selmeier equation found at https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
        around 1 micron.
    

        Parameters
        -------
        variable
        parameter = "wavelength" [nm] or "omega" [rad Hz]
         
    
        Returns
        -------
        value for refractive index of fused silica at wavelength.
        """
        print("In the function")
        print(parameter.lower())
        if parameter.lower() not in ["wavelength", "omega"]:
            raise NotImplementedError("Error: Input parameter needs to match 'wavelength' or 'omega'")
        if parameter.lower() == "omega":
            variable = 2 * np.pi * 3e17 / variable # Converts from omega to wavelength in nm, c is s.o.l in nm/s.
            
        wavelength = RefractiveIndex._nm2um(variable)
        return np.sqrt(1 + (0.6961663 * wavelength**2) / (wavelength**2  - 0.0684043**2) + (0.4079426 * wavelength**2) / (wavelength**2 - 0.1162414**2) + (0.8974794 * wavelength**2) / (wavelength**2 - 9.896161**2))
      
    def _deriv(f, wavelength, h=1e-6):
        return (1/(2  * h)) * (f(wavelength + h) - f(wavelength - h))
    
    def n_group(n, wavelength, h=1e-6):
        return n(wavelength) - wavelength * RefractiveIndex._deriv(n, wavelength, h)
    
# import matplotlib.pyplot as plt
# import numpy as np
# lambdas = np.linspace(1)