# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:56:45 2023

@author: Jack Morse

A class to help with general data analysis and manipulation.
"""
import numpy as np
import csv

class DataAnalysis:
    pass


class DataHandling:
    def write_csv(file_path, data, headers, preamble = []):        
        if len(headers) != len(data):
            print("Enter one header per coumn.")
            return
        zipped_data = zip(*data)
        data_list = list(zipped_data)
        # check file path name ends in csv
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if preamble is not []:
                writer.writerow(preamble)
            writer.writerow(headers)
            writer.writerows(data_list)
        
class Calculus:
    @staticmethod
    def derivative(func, x, h=1e-5):
        '''
        Computes the first derivative of a function 'func' at given x value,
        using the central difference method.

        Parameters
        -------
        func (function): function to differentiate, e.g. np.sin.
        x (float): point to evaluate the derivative.
        h (float): step size
         

        Returns
        -------
        The value for the derivative evaluated at x.
        '''
        return (func(x + h) - func(x - h))/(2*h)
