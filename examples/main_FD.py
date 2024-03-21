import random 
random.seed(0)
import numpy as np
import matplotlib.pyplot as plt

from udiff.smooth import smooth_polynomial
from udiff.differentiate import differentiator

#%% Generate concentration data
# ============================================================================
EXAMPLE = 0
nsamples = 15

if EXAMPLE == 0:
    X = np.linspace(1e-3, np.pi, nsamples)
    Y_real = X**2 + 3*np.exp(X)
    np.random.seed(0)
    noise = np.random.normal(0,0.1,len(Y_real))
    Y = Y_real * (1+noise)
    dY = 2*X + 3*np.exp(X)
    
elif EXAMPLE == 1:
    X = np.linspace(-6, 6, nsamples)
    Y_real = np.exp(X) / (np.exp(X) + 1)
    np.random.seed(0)
    noise = np.random.normal(0,0.1,len(Y_real))
    Y = Y_real * (1+noise)
    dY = np.exp(-X) / ((1 + np.exp(-X))**2)



#%% Forward finite difference
# ============================================================================
FDobj = differentiator(xdata = X, ydata = Y)
FDobj.differentiate(type_of_numerical_differentiation = 'FD')
FDobj.compare_calculated_and_real_derivatives(calc_diff=FDobj.y_diff,
                                                real_diff=dY[:-1],
                                                type_of_comp=['OVP', 'profile'])


#%% Total Variation Regularized Differentiation
# ============================================================================
TVRDobj = differentiator(xdata = X, ydata = Y)
TVRDobj.differentiate(type_of_numerical_differentiation = 'TVRD', alpha=1e-3, no_opt_steps=2000)
TVRDobj.compare_calculated_and_real_derivatives(calc_diff=TVRDobj.y_diff,
                                                real_diff=dY[:-1],
                                                type_of_comp=['OVP', 'profile'])


#%% Show plots
# ============================================================================
plt.show()

