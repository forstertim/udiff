import random 
random.seed(0)
import numpy as np
import matplotlib.pyplot as plt

from udiff.smooth import smooth_functionlibrary
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
INITIAL_GUESS = {   'exp': None, 
                    'poly2': None, 
                    'poly4': [0, 1, 1, 1, 1], 
                    'sigmoid': [1, 1, 1, 1],
                    'custom': {0: ['x**a + b*exp(x)', [2, 3]]}}
obj = smooth_functionlibrary(X, Y, scaling=None, initial_guess=INITIAL_GUESS)
obj.print_function_library()
obj.fit(no_warnings=False, randomize_initial_guess=5)
obj.plot_all_fits()
obj.plot_best_fit()


#%% Differentiation
# ============================================================================
diffobj = differentiator(obj)
diffobj.differentiate()


#%% Show plots
# ============================================================================
plt.show()
