import random 
random.seed(0)
import numpy as np
import matplotlib.pyplot as plt

from udiff.smooth import smooth_bms
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



#%% Fitting
# ============================================================================
obj = smooth_bms(x=X, y=Y, scaling=False)
obj.fit_bms(nsteps=150, maxtime=100, minr2=0.98, savename='bms_series1')
obj.plot_fit()


#%% Differentiation
# ============================================================================
diffobj = differentiator(obj)
diffobj.differentiate()
diffobj.compare_calculated_and_real_derivatives(calc_diff=diffobj.y_diff,
                                                real_diff=dY,
                                                type_of_comp=['OVP', 'profile'])


#%% Show plots
# ============================================================================
plt.show()
