import os
import random 
random.seed(0)
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint  
from sklearn.metrics import mean_squared_error

from udiff.smooth import smooth_bms
from udiff.differentiate import differentiator

#%% Generate concentration data
# ============================================================================
class datagen():

    # ------------------------------------------------------------------------
    def __init__(self, 
                 y0 = np.array([0.1, 60, 0]),
                 tspan = np.linspace(0, 80, 20)):
        self.y0 = y0
        self.tspan = tspan

    # ------------------------------------------------------------------------
    def solve_ODE_model(self):
        '''Solve the ODE model for the given sample and time span. \
            Returns the derivatives (dydt) as an array of shape [n,] and the runtime of the ODE solver.
        '''

        # Simulate the reactor operation until the selected time tf
        self.y = odeint(func=self.ODEmodel, y0=self.y0, t=self.tspan)

    # ------------------------------------------------------------------------
    def ODEmodel(self, y, t):
        '''ODE model for batch fermentation. Literature: Turton, Shaeiwtz, Bhattacharyya, Whiting \
            "Analysis, synthesis and design of chemical processes", Prentice Hall. 2018.
        '''

        # Variables  
        X = y[0]
        S = y[1]
        P = y[2]

        # Parameters
        mu_max = 0.25; 	#h^-1
        K_S = 105.4;    #kg/m^3
        Y_XS = 0.07;    #[-]
        Y_PS = 0.167;   #[-]

        KXmu = 121.8669;#g/L constant for inhibition of biomass growth caused by higher cell densities

        T = 273 + 35;    #K
        R = 0.0083145;  #kJ/(K*mol) universial gas constant

        k1_ = 130.0307;  #[-] constant for activation of biomass growth
        E1_ = 12.4321; 	#kJ/mol activation enthalpy for biomass growth
        k2_ = 3.8343e48; #[-] constant for inactivation of biomass growth
        E2_ = 298.5476;	#kJ/mol inactivation enthalpy for biomass growth
        
        # Define temperature dependency of the rate constants
        k1 = k1_ * np.exp(-E1_ /(R*T))
        k2 = k2_ * np.exp(-E2_ /(R*T))

        # Calculate specific growth rate of the biomass
        mu = (mu_max*S)/(K_S+S) * k1/(1+k2) * (1-(X/(KXmu+X)))

        # Calculate consumption of substrate
        sigma = -(1/Y_XS)*mu

        # Calculate specific production rate of the protein
        pi = Y_PS/Y_XS*mu

        # Vectorization of rates
        rate = np.hstack((mu.reshape(-1,1), sigma.reshape(-1,1), pi.reshape(-1,1)))        

        # ODEs of biomass, volume and product  
        dydt = rate * X.reshape(-1,1)

        # Return
        return dydt.reshape(-1,)
    
    # ------------------------------------------------------------------------
    def addnoise_per_species(self, percentage=5):
        '''Uses some ground truth data and adds some noise to it. 
        '''
        self.y_noisy = np.zeros((self.y.shape[0], 0))
        for spec_id in range(self.y.shape[1]):
            rmse = mean_squared_error(self.y[:, spec_id], np.zeros(self.y[:, spec_id].shape), squared=False)
            y_noisy_spec = self.y[:, spec_id] + np.random.normal(0, rmse / 100.0 * percentage, self.y[:, spec_id].shape)
            self.y_noisy = np.hstack((self.y_noisy, y_noisy_spec.reshape(-1,1)))
    
    # ----------------------------------------
    def evaluate_true_derivatives(self):
        '''Evaluate the true derivatives of the generated data.
        '''
        self.y_true_diff = np.zeros((0, self.y.shape[1]))
        for t_id, t in enumerate(self.tspan):
            self.y_true_diff = np.vstack((self.y_true_diff, self.ODEmodel(y=self.y[t_id, :], t=t).reshape(1,-1)))

    # ------------------------------------------------------------------------
    def plot_profiles_and_derivatives(  self, 
                                        show:bool=True,
                                        save=False, 
                                        save_figure_directory='./Figures', 
                                        save_figure_exensions:list=['png']):
        '''Plot batch that was simulated.
        '''       

        # Create figure
        fig, ax = plt.subplots(ncols=self.y.shape[1], nrows=2, figsize=(10,5))

        # Plot data
        for spec_id in range(self.y.shape[1]): 
            # Profiles
            ax[0,spec_id].plot(self.tspan, self.y[:, spec_id], color='black', marker='', linestyle='--')            # True concentrations
            ax[0,spec_id].plot(self.tspan, self.y_noisy[:, spec_id], color='black', marker='o', linestyle='')       # Noisy concentrations
            ax[0,spec_id].plot(self.tspan, self.y_smooth[:, spec_id], color='r', marker='', linestyle='-')          # Smoothed concentrations
            # Labels
            ax[0,spec_id].set_xlabel('Time')
            ax[0,spec_id].set_ylabel(f'Concentration X$_{spec_id}$')
            # Species
            ax[1,spec_id].plot(self.tspan, self.y_true_diff[:, spec_id], color='black', marker='', linestyle='--')  # True derivatives
            ax[1,spec_id].plot(self.tspan, self.y_diff[:, spec_id], color='red', marker='x', markersize=4, linestyle='-')         # Estimated derivatives by BMS 
            ax[1,spec_id].plot(self.x_diff_fd, self.y_diff_fd[:, spec_id], color='b', marker='d', markersize=4, linestyle='-')         # Estimated derivatives by FD
            # Labels
            ax[1,spec_id].set_xlabel('Time')
            ax[1,spec_id].set_ylabel(f'Derivative X$_{spec_id}$')

        # Layout
        plt.tight_layout()
        
        # Save figure if required
        figname = 'Bioproces_generated_data'
        self.save_figure(save, figname, save_figure_directory, save_figure_exensions)

        # Show
        if show:
            plt.show()
    
    # ----------------------------------------
    def save_figure(self, 
                    save:bool=False, 
                    figure_name:str='figure', 
                    savedirectory:str='./Figures', 
                    save_figure_exensions:list=['svg','png']):
        """Saves a figure.

        Args:
            save (bool): Boolean indicating whether the figure should be saved. Defaults to False
            figure_name (str): Name of the figure. Defaults to 'figure'.
            savedirectory (str): Directory in which the figure should be saved. Defaults to './figures'.
            save_figure_exensions (list): List of file extensions in which the figure \
                should be saved. Defaults to ['svg','png'].
        """
    
        if save:
            print(f'[+] Saving figure:')
            if isinstance(save_figure_exensions, list):
                figure_extension_list = save_figure_exensions
            else:
                raise ValueError('[-] The indicated file extension for figures needs to be a list!')
            for figure_extension in figure_extension_list:
                savepath = os.path.join(savedirectory, figure_name+'.'+figure_extension)
                plt.savefig(savepath)
                print(f'\t->{figure_extension}: {savepath}')
        else:
            print(f'[+] Figures not saved.')


data = datagen()
data.solve_ODE_model()
data.addnoise_per_species()
data.evaluate_true_derivatives()


#%% Fitting
# ============================================================================
# Create empty arrray for fitted profiles
data.y_smooth = np.zeros(data.y.shape)
data.y_diff = np.zeros(data.y.shape)
data.x_diff_fd = np.zeros((data.y.shape[0]-1, data.y.shape[1]))
data.y_diff_fd= np.zeros((data.y.shape[0]-1, data.y.shape[1]))

# Fit profiles 
for spec_id in range(data.y.shape[1]):
    # Each species individually
    X = data.tspan
    Y = data.y_noisy[:, spec_id]
    # Create new object
    obj = smooth_bms(x=X, y=Y, scaling=False)
    # Fit profiles
    obj.fit_bms(nsteps=1e4, maxtime=3.6e3, minr2=0.999, show_update=True, update_every_n_seconds=200) 
    # Store data
    data.y_smooth[:, spec_id] = obj.y_smooth
    # Differentiate algebraic equation
    diffobj = differentiator(obj)
    diffobj.differentiate()
    # Store data
    data.y_diff[:, spec_id] = diffobj.y_diff
    # Calculate finite differences for comparison
    y_der_FD = []
    x_der_FD = []
    dt = (data.tspan[1] - data.tspan[0])/2
    for t_ in range(len(data.tspan)-1):
        x_der_FD.append(data.tspan[t_] + dt)
        y_der_FD.append((Y[t_+1] - Y[t_])/(data.tspan[t_+1] - data.tspan[t_]))
    data.x_diff_fd[:, spec_id] = np.array(x_der_FD)
    data.y_diff_fd[:, spec_id] = np.array(y_der_FD)


#%% Show plots
# ============================================================================
data.plot_profiles_and_derivatives(save=True)
plt.show()
