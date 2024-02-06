
def BisectionMethod(func, interval, TOL = 1e-6, NMAX = 1e3):
    import math
    sign = lambda x: math.copysign(1, x)
    N = 1
    a = interval[0]
    b = interval[1]
    while N <= NMAX:
        c = (a + b) / 2
        if func(c) == 0 or (b - a) / 2 < TOL:
            return c
        N += 1
        if sign(func(c)) == sign(func(a)):
            a = c
        else:
            b = c
    print("Method failed: Max number of steps exceeded.")
        
def polynomial(x):
    return x**3 - x - 2

# Define the function we wish to minimise. We have two functions: 
#               omega_sig = 2 * omega_pump - omega_idler
#               
print(BisectionMethod(polynomial, [1, 2]))