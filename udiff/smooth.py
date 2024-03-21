import sys
import random 
random.seed(0)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sympy import sympify, lambdify, symbols, Symbol
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import socket
import time
import pickle
from .BMS_utils import BMS_instance

#%% MODEL FITTING MODULE
class smooth_functionlibrary:
    '''
    This class is used to create an object that fits a function library to some observed data points.
    The function library should be customizable. Creates the instance and populates the object with the necessary data and information
    about the function libraries. It scales the data (if required) and calls the library.
    The 'type of smooth' defines that the fitting is done by a sympy expression (this info is later on 
    used for the function differentiation).

    Args:
        x (array, required): Abszissa data (x-axis)
        y (array, required): Concentration data (y-axis)
        scaling (str, optional): 'standardscaling' or 'minmaxscaling' if corresponding scaling should be applied
        initial_guess (dict, optional): Dictionary with the initial guesses of the parameters in the user-defined functions of the library
    '''

    # ===================================================================================
    def __init__(self,
                    x,
                    y,
                    scaling : str = '',
                    initial_guess : dict = {'exp': [0, 0, 0, 0]}) -> None:
        """Initializer function.
        """

        self.x = x
        self.y = y
        self.scaling = scaling
        self.initial_guess = initial_guess
        self.allowed_functions = list(initial_guess.keys())        
        self.type_of_smooth = 'sympy' # with this tag, we can later on easily check how to differentiate the data

        # Scale data if required
        self.scale_y()

        # Generate desired function library
        self.function_library()

    # ===================================================================================
    def scale_y(self) -> None:
        '''Scales the data if required. No inputs required

        Stores:
            :scaler: Scaler object
            :y (array): Scaled y-data
        '''
        old_shape = self.y.shape
        if self.scaling == 'standardscaling':
            self.scaler = StandardScaler()
            self.y = self.scaler.fit_transform(self.y.reshape(-1,1))
        elif self.scaling == 'minmaxscaling':
            self.scaler = MinMaxScaler()
            self.y = self.scaler.fit_transform(self.y.reshape(-1,1))
        else:
            self.scaler = 'no scaling'
        self.y = self.y.reshape(old_shape)

    # ===================================================================================
    def function_library(self) -> None:
        '''Function library. The user can define new functions here and the alogrithm will fit those functions
        to the data. To fit the desired function, the initial guessess for the parameters should be handed 
        over in the __init__ method. No inputs, everything is defined by the __init__ method! 
        The method then converts all the functions to sympy expressions and stores them in the instance.
        '''
        a,b,c,d,e,f,g,x = symbols('a b c d e f g x')
        from sympy import exp, cos, sin, log

        # sympy expressions
        sympy_functions = {}
        initial_guess = {}
        function_counter = 0

        # Check all the standard functions
        # -------------------------------------------------------------------------------
        for funcnumber, func in enumerate(self.allowed_functions):
            if 'exp' == func:
                sympy_functions[funcnumber] = a + b*exp(c + d*x)
            if 'poly1' == func:
                sympy_functions[funcnumber] = a + b*x
            if 'poly2' == func:
                sympy_functions[funcnumber] = a + b*x + c*(x**2)
            if 'poly3' == func:
                sympy_functions[funcnumber] = a + b*x + c*(x**2) + d*(x**3)
            if 'poly4' == func:
                sympy_functions[funcnumber] = a + b*x + c*(x**2) + d*(x**3) + e*(x**4)
            if 'sigmoid' == func:
                sympy_functions[funcnumber] = a * (exp(b*x) / (exp(c*x) + d))
            
            # Store initial guesses 
            if self.initial_guess[self.allowed_functions[funcnumber]] is None:
                initial_guess[funcnumber] = None
            else:
                initial_guess[funcnumber] = self.initial_guess[self.allowed_functions[funcnumber]]
 
        # Check all the custom functions
        # -------------------------------------------------------------------------------
        new_func_number = len(sympy_functions.keys())
        if 'custom' in self.allowed_functions:
            # Check how many custom functions are given
            list_custom_func_ids = list(self.initial_guess['custom'].keys())
            # Iterate through all custom functions
            for custom_func_id in list_custom_func_ids:
                # Store expression
                sympy_functions[new_func_number] = sympify(self.initial_guess['custom'][custom_func_id][0])
                # Store initial guess to it
                if self.initial_guess['custom'][custom_func_id][1] is None:
                    initial_guess[new_func_number] = None
                else:
                    initial_guess[new_func_number] = self.initial_guess['custom'][custom_func_id][1]
                # Increase function number
                new_func_number += 1

        # Store functions
        self.sympy_functions = sympy_functions
        self.initial_guess = initial_guess  # Overwrite initial guess dict with keys to be integers
        self.lambdify_functions()

    # ===================================================================================
    def lambdify_functions(self) -> None:
        '''Method to convert the sympy expressions to lambda-functions in order to evaluate them later on.
        '''
        func = {}
        self.n_params = {}
        self.param_list = {}
        for i in range(len(self.sympy_functions.keys())):
            symbs = list(self.sympy_functions[i].atoms())               # Get sympy symbols
            symbs = [x for x in symbs if not sympify(x).is_integer]     # Only consider symbols, not "power-of-2" (no integers)
            symbs = [str(x) for x in symbs]                             # Convert symbols to strings 
            symbs.sort()                                                # Sort strings
            symbs.remove('x')                                           # Remove 'x' from the list, since we only want the parameters in ordered sequence
            self.n_params[i] = len(symbs)                               # Store the number of parameters (without the variable 'x')
            self.param_list[i] = symbs                                  # Store the parameters of each function
            symb_list = ['x']+symbs                                     # Add 'x' (the varible) as the first element in the list
            symbs = [Symbol(x) for x in symb_list]                      # Convert the whole list again back to symbols
            func[i] = lambdify(symbs, self.sympy_functions[i])          # Lambdify (first element has to be the variable, the rest are the parameters)
        self.functions = func

    # ===================================================================================
    def print_function_library(self) -> None:
        '''Prints the functions to the console.
        '''
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Index", "Function"]
        for func in range(len(self.functions.keys())):
            table.add_row([f'{func}', str(self.sympy_functions[func])])
        print(table)

    # ===================================================================================
    def plot_all_fits(self) -> None:
        '''Plots all the functions that were fitted together with the observed data. 
        Also, a table is printed with the functions.
        '''
        plt.figure()
        plt.plot(self.x, self.y, 
                 'bo',
                 label = 'Observed')
        for func in range(len(self.functions.keys())):
            if self.yfit[func] is not None:
                plt.plot(self.x, self.yfit[func], 
                        label = f'Fit-{func}')
        plt.legend(loc='best', frameon=False)

        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Index", "Function"]
        for func in range(len(self.functions.keys())):
            table.add_row([f'{func}', str(self.sympy_functions[func])])
        print(table)

    # ===================================================================================
    def plot_best_fit(self) -> None:
        '''The best-performing function is plotted together with the observed data.
        '''
        plt.figure()
        plt.plot(self.x, self.y, 
                 'bo',
                 label = 'Observed')
        plt.plot(self.x_smooth, self.y_smooth, label = f'Best: {self.best_func}')
        plt.legend(loc='best', frameon=False)

    # ===================================================================================
    def fit(self,
            no_warnings : bool = True,
            randomize_initial_guess : int = 0) -> None:
        '''All the functions are fit to the observed data.
        
        Args:
            no_warnings (bool, optional): Decide if warnings of the curve fitting should be printed to \
                the console or not (true or false, default to true - no warnings)
            randomize_initial_guess (int, optional): In case the curve fitting fails, the initial guesses are multiplied \
                with a random number and the curve fit is executed again. Hand over an integer >=0.
        '''
        
        if no_warnings:
            import warnings
            warnings.filterwarnings("ignore")

        params = {}
        for func in range(len(self.functions.keys())):
            # If initial guess is given, use this in curve_fit
            if self.initial_guess[func] is not None:
                if self.n_params[func] == len(self.initial_guess[func]):
                    try:
                        params[func], _ = curve_fit(f=self.functions[func], xdata=self.x, ydata=self.y, p0=self.initial_guess[func])
                    except:
                        for init in range(0, randomize_initial_guess):
                            try:
                                print(f'[+] Performing randomization {init+1}/{randomize_initial_guess} of initial guess of function {func}.')
                                randnumber = []
                                for i_ in range(len(self.initial_guess[func])):
                                    randnumber.append(np.random.normal(self.initial_guess[func][i_], 10))
                                self.initial_guess[func] = ((np.array(self.initial_guess[func]) + 1e-10) * np.array(randnumber)).tolist()
                                params[func], _ = curve_fit(f=self.functions[func], xdata=self.x, ydata=self.y, p0=self.initial_guess[func])
                                print(f'\t Successful!')
                                break
                            except:
                                print(f'\t Not successful!')
                                pass
                else:
                    sys.exit(f'[-] ABORTED! Incorrect number of initial guess elements in function with index {func} (>>{str(self.sympy_functions[func])}<< ---> given {len(self.initial_guess[func])} guess elements, but required are {self.n_params[func]})!')
            # If no initial guess is given, use default of curve_fit
            else:
                try:
                    params[func], _ = curve_fit(f=self.functions[func], xdata=self.x, ydata=self.y)
                except:
                    params[func] = None 
        self.params = params

        # Predict with fitted functions
        self.predict_all()

        # Print a report with the found fitness metric and the parameters
        self.print_report()

        # Choose best model and save it to object
        # -> x_smooth: x-data of smooth
        # -> y_smooth: y-data of smooth
        # -> smoothing_model: smoothing function obtained (sympy expression)
        self.choose_best()

    # ===================================================================================
    def fit_metric(self, 
                   observed, 
                   predicted) -> None:
        '''The mean squared error (MSE) is calculated.

        Args:
            observed (array): Observed data.
            predicted (array): Predicted data.
        '''
        return mean_squared_error(observed, predicted)

    # ===================================================================================
    def predict_all(self) -> None:
        '''The predictions of all model fittings are calculated. \
        Then, the fitness (error metric) is calculated and stored.
        '''
        yfit = {}
        fitness = {}
        for func in range(len(self.functions.keys())):
            if self.params[func] is not None:
                yfit[func] = self.functions[func](self.x, *self.params[func])
                if len(yfit[func]) != len(self.y) or np.isnan(yfit[func]).any():
                    fitness[func] = None
                else:
                    fitness[func] = self.fit_metric(self.y, yfit[func])
            else:
                yfit[func] = None
                fitness[func] = None
        self.yfit = yfit 
        self.fitness = fitness

    # ===================================================================================
    def choose_best(self) -> None:
        '''The best function is searched and stored in the attribute 'model'.
        '''
        best_func = self.fitness[0]
        best_func_ind = 0
        for ind_i, i in enumerate(list(self.fitness.keys())[1:]):
            if best_func > self.fitness[ind_i]:
                best_func = self.fitness[ind_i]
                best_func_ind = i
        self.y_smooth = self.yfit[best_func_ind]
        self.x_smooth = self.x
        self.best_func = self.sympy_functions[best_func_ind]
        self.best_param = self.params[best_func_ind]
        print(f'[+] Best function index: {best_func_ind}')
        print(f'[+] Best function fitness: {best_func:.3e}')
        func = self.best_func
        for i, par_ in enumerate(self.param_list[best_func_ind]):
            func = func.subs(par_, self.best_param[i])
        self.model = func

    # ===================================================================================
    def print_report(self) -> None:
        '''The full results are printed to the console (functions chosen, error metrics, parameter estimates).
        '''

        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Index", "Fit metric (RMSE)", "Function", "Identified parameters"]

        for func in range(len(self.functions.keys())):
            if self.fitness[func] is not None:
                params = [f'{x:.3f}' for x in self.params[func]]
                table.add_row([f'{func}', f'{self.fitness[func]:.3e}', f'{str(self.sympy_functions[func])}', f'{*params,}'])
            else:
                table.add_row([f'{func}', 'None', f'{str(self.sympy_functions[func])}', 'None'])
        print(table)


#%% POLYNOMIAL FITTING MODULE
class smooth_polynomial:
    '''
    This class is used to create an object that fits a polynomial function to some observed data points. 
    Creates the instance and populates the object with the necessary data and information
    about the function libraries. It scales the data (if required) and calls the library.
    The 'type of smooth' defines that the fitting is by a polynomial fit (this info is later on 
    used for the function differentiation).

    Args:
        x (array, required): Abszissa data (x-axis)
        y (array, required): Concentration data (y-axis)
        scaling (str, optional): 'standardscaling' or 'minmaxscaling' if corresponding scaling should be applied
        initial_guess (dict, optional): Dictionary with the initial guesses of the parameters in the user-defined functions of the library
    '''

    # ===================================================================================
    def __init__(   self,
                    x,
                    y,
                    scaling = False) -> None:
        """Initializer function.
        """
        
        self.x = x
        self.y = y
        self.scaling = scaling
        self.type_of_smooth = 'polynomial' # with this tag, we can later on easily check how to differentiate the data

        # Scale data if required
        self.scale_y()

    # ===================================================================================
    def scale_y(self) -> None:
        '''Scales the data if required.
        '''
        old_shape = self.y.shape
        if self.scaling == 'standardscaling':
            self.scaler = StandardScaler()
            self.y = self.scaler.fit_transform(self.y.reshape(-1,1))
        elif self.scaling == 'minmaxscaling':
            self.scaler = MinMaxScaler()
            self.y = self.scaler.fit_transform(self.y.reshape(-1,1))
        else:
            self.scaler = 'no scaling'
        self.y = self.y.reshape(old_shape)

    # ===================================================================================
    def fit_polynomial(self, maxdeg=8) -> None:
        '''Different polynomials are fit to the observed data. The number of polynomials is indicated
        by the maximum degree that is defined by the user (i.e., if maxdeg=2, one polynomial of first
        order and another polynomial of second order is fit through the data).
        The Bayesian Information Criterion (BIC) is then calculated and the polynomial with the lowest
        BIC is chosen to be the best model. The identified coefficients are then saved in the 
        attribute 'model'.

        Args:
            maxdeg (int, optional): Maximum degree of the polynomial
        '''
        # Try out different degrees and choose best according to best criterion
        self.maxdeg = maxdeg
        DEG = np.arange(1,self.maxdeg)
        self.best_BIC = 1e16
        self.best_ssr = 1e16
        self.best_p = None
        self.best_deg = None
        for deg in DEG:
            p, ssr, _, _, _ = np.polyfit(self.x, self.y, deg, full=True)
            BIC = len(self.y)*np.log(ssr/len(self.y))+len(p)*np.log(len(self.y))
            if BIC < self.best_BIC:
                self.best_deg = deg
                self.best_p = p
                self.best_ssr = ssr
                self.best_BIC = BIC

        # Evaluate polynomial at t_batch
        # -> x_smooth: x-data of smooth
        # -> y_smooth: y-data of smooth
        # -> smoothing_model: smoothing function obtained (sympy expression)
        self.y_smooth = np.polyval(self.best_p, self.x)
        self.x_smooth = self.x
        self.model = {'deg': self.best_deg, 'p': self.best_p}
        
    # ===================================================================================
    def plot_fit(self):
        '''Visualize the best fit found.
        '''
        plt.figure()
        plt.plot(self.x, self.y, 
                 'bo',
                 label = 'Observed')
        plt.plot(self.x_smooth, self.y_smooth, label = f'd={self.best_deg}')
        plt.legend(loc='best', frameon=False)


#%% SYMBOLIC FITTING MODULE
class smooth_bms:
    '''This class is used to create an object that fits a BMS model to some observed data points.
    The BMS is a symbolic regression method published by: Roger Guimerà et al., A Bayesian machine \
        scientist to aid in the solution of challenging scientific problems.Sci. Adv.6, (2020) \
        https://www.science.org/doi/10.1126/sciadv.aav6971

    Creates the instance and populates the object with the necessary data and information
    about the function libraries. It scales the data (if required) and calls the library.        
    The 'type of smooth' defines that the fitting is done by a sympy expression (this info is later on 
    used for the function differentiation).

    Args:
        x (array, required): Abszissa data (x-axis)
        y (array, required): Concentration data (y-axis)
        scaling (str, optional): 'standardscaling' or 'minmaxscaling' if corresponding scaling should be applied 
        npar (int, optional): Number of parameters used for the BMS fit. The univariate (no of variables=1) has \
            npar options from 1 to 9.
    '''

    # ===================================================================================
    def __init__(   self,
                    x,
                    y,
                    scaling = False,
                    npar : int = None) -> None:
        """Initializer function.
        """

        self.x = x
        self.y = y
        self.scaling = scaling
        self.type_of_smooth = 'sympy' # with this tag, we can later on easily check how to differentiate the data
        self.npar = npar

        # Store data in dataframe
        self.traindata = pd.DataFrame(x, columns=['x'])
        self.traindata['y'] = y

        # Scale data if required
        self.scale_y()

    # ===================================================================================
    def scale_y(self) -> None:
        '''Scales the data if required.
        '''
        old_shape = self.y.shape
        if self.scaling == 'standardscaling':
            self.scaler = StandardScaler()
            self.y = self.scaler.fit_transform(self.y.reshape(-1,1))
        elif self.scaling == 'minmaxscaling':
            self.scaler = MinMaxScaler()
            self.y = self.scaler.fit_transform(self.y.reshape(-1,1))
        else:
            self.scaler = 'no scaling'
        self.y = self.y.reshape(old_shape)

    # ===================================================================================
    def fit_bms(self,
                nsteps : int = 2000,
                maxtime : int = None,
                minr2 : float = None,
                ops : dict = None,
                nsaves : str = 'final', 
                mode : str = 'train',
                num_dummy_var : int = 0,
                no_save : bool = True,
                show_update : bool = False,
                update_every_n_seconds : float = 30,
                savename : str = 'bms') -> None:
        '''A BMS model is fit to the data points.

        Args:
            nsteps (int, optional): Number of MCMC (Markov-Chain Monte Carlo) steps. Default to 2000.
            nsaves (int, optional): List of integers. If a list of integers is indicated, the BMS model after those steps is saved.\
                Can also be 'final' or an empty list to only save the final model at the indicated number of MCMC ('nsteps').
            maxtime (float, optional): Fitting stops if the indicated number of seconds are reached. Default to None. \
                If none given, nsteps are chosen as stopping criteria.
            minr2 (float, optional): Fitting stops if the indicated coefficient of determination (R2-value) is reached. Default to None.\
                If none given, either time or nsteps are chosen as stopping criteria.
            ops (dict, optional): Dictionary of chosen operations for symbolic regression. Defaults are set below.
            mode (str, optional): Indicate if a training or loading of the BMS model should be done. Currently, a reload of a saved model is not available yet.
            num_dummy_var (int, optional): Number of dummy variables (columns of zeros) that should be added. Currently, this function is dedicated to univariate fitting \
                so a dummy variable is not implemented yet.
            no_save (bool, optional): Define if the final BMS model is saved or not. Default to not-saved (True).
            show_update (bool, optional): Define if updates about the fit are shown after some iterations. Default to False.
            update_every_n_seconds (float, optional): Define the time in seconds after which an update is shown. Default to 30 seconds.
            savename (str, optional): Filename of the saved BMS model.
        '''

        # Define hyperparameters
        if nsaves == 'final' or nsaves == []:
            nsaves = [nsteps]
        if not isinstance(nsaves, list):
            raise ValueError('[-] >>nsaves<< needs to be a list!')
        if np.max(nsaves) > nsteps:
            raise ValueError(f'[-] The largest intermediate-save {np.max(nsaves)} cannot be larger than the maximum training MCMC steps ({nsteps})!')

        # Define used operations
        if ops is None:
            OPS = {
                'sin': 1,
                'cos': 1,
                'tan': 1,
                'exp': 1,
                'log': 1,
                # 'sinh' : 1,
                # 'cosh' : 1,
                # 'tanh' : 1,
                'pow2' : 1,
                'pow3' : 1,
                # 'abs'  : 1,
                'sqrt' : 1,
                # 'fac' : 1,
                '-' : 1,
                '+' : 2,
                '*' : 2,
                '/' : 2,
                '**' : 2,
                }
        else:
            OPS = ops

        # Define data folder
        bms_model_save_folder_path = r'.'

        # Load instance of BMS
        bms = BMS_instance( experiment = 'timeseries_fit', 
                            noutputs = 1, 
                            chosen_output = 1, 
                            scaling = None, # done separately before -> this scaling feature of the BMS here is not used
                            data_path = None,
                            prior_folder = r'./BMS_requirements',
                            ops = OPS,
                            traindata = self.traindata,
                            no_save = no_save,
                            num_dummy_var = num_dummy_var,
                            npar = self.npar)

        # Check host machine on which BMS is trained
        bms.hostmachine_training = socket.gethostname()

        # Record time
        starttime = time.time()

        # Run BMS if MCMC steps are stopping criteria
        print('[+] Start BMS fit')
        if maxtime is None and minr2 is None:
            print(f'\t[!] Chose only number of MCMC steps as stopping criterion!')
            bms.run_BMS(mcmcsteps = nsteps, 
                        save_distance = nsaves, 
                        save_folder_path = bms_model_save_folder_path)

        else:
            # Print some infos
            if maxtime is not None:
                if not isinstance(maxtime, float) and not isinstance(maxtime, int):
                    raise ValueError('[-] The maximum runtime for the BMS fit needs to be of type int or float!')
                print(f'\t[!] Chose {maxtime:.0f} seconds as maximum runtime for BMS fit!')
            if minr2 is not None:
                if not isinstance(minr2, float) or minr2 > 1:
                    raise ValueError('[-] The minimum R2 value needs to be a float value between [-inf, 1)!')
                print(f'\t[+] Chose {minr2:.4f} as minimum R2 value for BMS fit!')
            
            # Start the fit in a loop
            # Use the option "initialize=False" after the first iteration to stop the BMS from recreating a new set of symbolic trees
            increment_nsteps = 10
            cum_nsteps = 0
            initialize_mdl = True
            print_update_time = 0
            print_update_starttime = time.time()
                
            # Start fits
            while True:
                cum_nsteps += increment_nsteps
                bms.run_BMS(mcmcsteps = increment_nsteps, 
                            save_distance = nsaves, 
                            save_folder_path = bms_model_save_folder_path,
                            initialize=initialize_mdl) 
                self.r2(overwrite_y_predictions=bms.mdl_model.predict(bms.x).values)
                current_r2 = self.r2_train
                current_time = time.time() - starttime

                # After the first iteration, turn of initialization of the trees
                if initialize_mdl:
                    initialize_mdl = False

                # Stop if nsteps are reached
                if cum_nsteps >= nsteps:
                    print(f'\t[!] Met criterion of max MCMC steps: {cum_nsteps:.0f} >= {nsteps:.0f}. Terminating!')
                    print(f'\t\t-> Time needed: {current_time:.0f} s')
                    print(f'\t\t-> R2: {current_r2:.4f}')
                    break

                # Stop if maxtime is reached
                if current_time >= maxtime:
                    print(f'\t[!] Met criterion of max fitting-time: {current_time:.0f} s >= {maxtime:.0f} s. Terminating!')
                    print(f'\t\t-> Executed MCMC steps: {cum_nsteps:.0f}')
                    print(f'\t\t-> R2: {current_r2:.4f}')
                    break
                
                # Stop if minimum R2 value is reached
                if current_r2 >= minr2:
                    print(f'\t[!] Met criterion of min R2: {current_r2:.4f} >= {minr2:.4f} (required). Terminating!')
                    print(f'\t\t-> Executed MCMC steps: {cum_nsteps:.0f}')
                    print(f'\t\t-> Time needed: {current_time:.0f} s')
                    break

                # Display some updates from time to time
                if show_update:
                    if print_update_time >= update_every_n_seconds:
                        print_update_time = 0
                        print_update_starttime = time.time()
                        print(f'\t[!] Update: Fitting BMS to data. Current MCMC steps: {cum_nsteps:.0f}. Current runtime: {current_time:.0f} seconds (max: {maxtime:.0f}). Current R2: {current_r2:.4f} (required: {minr2:.4f}).')
                    else:
                        print_update_time = time.time() - print_update_starttime

        bms.traintime = time.time() - starttime # saves last model trainingtime

        # If intermediate trainsteps are given and intermediate models were saved, delete this info for the final model
        if hasattr(bms, 'intermediate_trainsteps'):
            delattr(bms, 'intermediate_trainsteps')

        # Save BMS instance
        if not no_save:
            with open(bms_model_save_folder_path + '/' + r'{}.pkl'.format(savename), 'wb') as outp:
                pickle.dump(bms, outp, pickle.HIGHEST_PROTOCOL)

        # Store BMS
        self.bms = bms

        # Get list of parameters and variables
        parlist = self.bms.mdl_model.parameters
        varlist = self.bms.mdl_model.variables

        # Conert the BMS to a symbolic expression and remove the "_" signs
        from sympy.parsing.sympy_parser import parse_expr
        bms_string = str(self.bms.mdl_model).replace('_', '')
        bms_expression = parse_expr(bms_string)

        # Get parameter values (found in model)
        parvalues = self.bms.mdl_model.par_values['d0']

        # Insert regressed parameter values into the model
        for i in range(0, len(parlist)):
            parval = parvalues[parlist[i]]  # get parameter value
            bms_expression = bms_expression.subs(parlist[i].replace('_',''), parval)
        
        # Store things
        self.y_smooth = self.bms.mdl_model.predict(self.bms.x).values
        self.x_smooth = self.x
        self.model = bms_expression

    # ===================================================================================
    def r2(self, overwrite_y_predictions=None) -> None:
        ''' Calculation of the coefficient of determination (R2-value).

        Args:
            overwrite_y_predictions (array, optional): If the predictions should be overwritten, hand over the new predictions here.
        '''
        if overwrite_y_predictions is not None:
            predictions = overwrite_y_predictions
        else:
            predictions = self.y_smooth
        self.r2_train = r2_score(self.y, predictions)
        
    # ===================================================================================
    def plot_fit(self) -> None:
        '''Visualization of the BMS fit.
        '''
        plt.figure()
        plt.plot(self.x, self.y, 
                 'bo',
                 label = 'Observed')
        plt.plot(self.x_smooth, self.y_smooth, label = 'BMS')
        plt.legend(loc='best', frameon=False)


#%% POLYNOMIAL FITTING MODULE
class smooth_savitzky_golay:
    '''
    This class is used to create an object that applies a Savitzky-Golay (SG) filter to some observed data points.
    The scipy package is used for the SG filter.
    The 'type of smooth' defines that a SG filter was used (data is still discrete).

    Args:
        x (array, required): Abszissa data (x-axis)
        y (array, required): Concentration data (y-axis)
    '''

    # ===================================================================================
    def __init__(   self,
                    x,
                    y,
                    scaling = False) -> None:
        """Initializer function.
        """
        
        self.x = x
        self.y = y
        self.type_of_smooth = 'discrete' # with this tag, we can later on easily check how to differentiate the data


    # ===================================================================================
    def fit(self, maxwindowlength:int=8, maxorder:int=8, mode:str='interp',
            alpha:float=0.9) -> None:
        '''Different hyperparameters of the SG filter are applied. 
        The mean squared error (MSE) is then calculated and the combination with the lowest
        MSE is chosen to be the most useful fit. The smoothed discrete data is then saved in the object.

        Args:
            maxwindowlength (int, optional): Maximum window length of the SG filter. Default to 8.
            maxorder (int, optional): Maximum degree of the polynomial. Default to 8.
            mode (str, optional): Mode of the SG filter. Default to 'interp'.
            alpha (float, optiona): Weighting factor for the MSE and the variance. Default to 0.9.
        '''
        # Print a statement about alpha
        print(f'\t[!] Chose alpha={alpha:.2f} for the smoothing criterion. Might be useful to check different values.')
        # Try out different degrees and choose best according to best criterion
        self.maxorder = maxorder
        self.maxwindow = maxwindowlength
        ORD = np.arange(1,self.maxorder)
        WIN = np.arange(1,self.maxwindow)
        VAR_SCALING = sum([np.abs(self.y[i+1] - self.y[i]) for i in range(len(self.y)-1)])
        self.best_MSE = 1e16
        self.best_VAR = 1e16
        self.best_OBJ = 1e16
        self.best_order = None
        self.best_window = None
        for window_length in WIN:
            for order in ORD:
                # Check that the window length is less or equal to the number of data points
                if window_length > len(self.x):
                    break
                # Check that the polynomial order is less to the window length
                if order >= window_length:
                    break
                y_smooth = savgol_filter(self.y, window_length, order, mode=mode)
                MSE = mean_squared_error(self.y, y_smooth)
                VAR = sum([np.abs(y_smooth[i+1] - y_smooth[i]) for i in range(len(y_smooth)-1)])
                OBJ = alpha*MSE + (1-alpha)*(VAR/VAR_SCALING)
                if OBJ < self.best_OBJ: #MSE < self.best_MSE and VAR < self.best_VAR:
                    self.best_order = order
                    self.best_window = window_length
                    self.best_MSE = MSE
                    self.best_VAR = VAR
                    self.best_OBJ = OBJ
                    self.best_y_smooth = y_smooth

        # Store smoothed data
        self.x_smooth = self.x
        self.y_smooth = self.best_y_smooth

        
    # ===================================================================================
    def plot_fit(self):
        '''Visualize the best fit found.
        '''
        plt.figure()
        plt.plot(self.x, self.y, 
                 'bo',
                 label = 'Observed')
        plt.plot(self.x_smooth, self.y_smooth, label = f'w={self.best_window}, o={self.best_order}')
        plt.legend(loc='best', frameon=False)