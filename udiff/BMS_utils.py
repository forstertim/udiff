#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np
from copy import deepcopy
from .parallel import *
from .fit_prior import read_prior_par
import warnings
warnings.filterwarnings('ignore')
import pickle
import matplotlib.pyplot as plt
import time

class PriorError(Exception):
    pass
class ScalingError(Exception):
    pass

class BMS_instance:
    def __init__(self, ops:dict = None, experiment: str = None, noutputs:int = 1, chosen_output:int = 1, scaling:str = None, data_path:str = r"data.xslx",
                 prior_folder:str = r".", npar:int = None,
                 traindata = None,
                 no_save = True,
                 num_dummy_var:int = 0):
        """Initialize an instance of hte BMS class

        Args:
            experiment (str, optional): Name of the experiment if needed. Defaults to None.
            noutputs (int, optional): Number of outputs in the data. Defaults to 1.
            chosen_output (int, optional): Position of the chosen output, from 1 to noutputs. Defaults to 1.
            scaling (str, optional): Perform zscore scaling for inputs, outputs, both, or None. Defaults to None.
            data_path (str, optional): Path to the xlsx file. Defaults to r"data.xslx".
            ops (dict, optional): Dictionary with the valid operations and the children of each operation. Defaults to OPS.
            prior_folder (str, optional): Path to the prior folder. Defaults to r".".
            npar (int, optional): Number of parameters to consider. Defaults to None.
            num_dummy_var (int, optional): Number of dummy variable (column of zeros) that should be included. Defaults to 0.
        """     
        self.experiment = experiment
        self.traintime = [] # will be filled later on
        file_data = data_path
        import pathlib
        sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'priors'))
        self.prior_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), 'priors')
        self.ops = ops
        self.train = traindata
        self.num_dummy_var = num_dummy_var
        self.noutputs = noutputs
        self.ninputs  = self.train.shape[1] - self.noutputs
        self.init_prior(npar)
        self.init_tree(scaling = scaling, chosen_output=chosen_output)
        self.chosen_output = chosen_output
        self.no_save = no_save
        
    @staticmethod
    def load(load):
        """Static method to load pickled saved models.

        Args:
            load (str): Raw string that indicates the full path to the pickle (.pkl) file.

        Returns:
            BMS_instance: Loaded BMS instance
        """        
        with open(load, "rb") as input_file:
                return pickle.load(input_file)
        
    def init_prior(self, npar:int):
        """Initialize the prior considered for the BMS. It chooses the last element of a valid list of priors regarding
        the dimensionality of the input if no integer is passed.

        Args:
            npar (int): Numnber of parameters to consider

        Raises:
            PriorError: In case there is no priors available
        """    
        prior_folder = self.prior_folder
        prior_files  = os.listdir(prior_folder)
        self.valid_priors = [i for i in prior_files if ".nv{0}.".format(self.ninputs + self.num_dummy_var) in i]
        if npar:
            self.chosen_prior = [i for i in self.valid_priors if ".np{0}.".format(npar) in i][-1]
        else:
            self.chosen_prior = self.valid_priors[-1]
        if not self.chosen_prior:
            raise PriorError
        import re
        self.npar = int(re.search(r"np(\d+)", self.chosen_prior).group(1))
        self.prior_par = read_prior_par(prior_folder + "\\" + self.chosen_prior)
    
    def init_tree(self, chosen_output:int = 1, scaling:str = None):
        """Initializes the parallel BMS tree for the chosen output. Also applies z-score scaling if the option is selected.

        Args:
            chosen_output (int, optional): Position of the chosen output, from 1 to noutputs. Defaults to 1.
            scaling (str, optional): Perform zscore scaling for inputs, outputs, both, or None. Defaults to None.

        Raises:
            ScalingError: Wrong input to scaling. Options are 'inputs', 'outputs', 'both', and None.
        """        
        self.x  = self.train.iloc[:,:self.ninputs].copy()
        self.x.reset_index(inplace=True, drop = True)
        self.y  = self.train.iloc[:, self.ninputs + chosen_output - 1].copy()
        self.y  = pd.Series(list(self.y))
        self.original_columns = self.x.columns
        if len(self.x.columns) > 1:
            raise ValueError('[-] This module is only designed to fit a univariate function!')
        self.x.columns = ["x"]
        if self.num_dummy_var > 0:
            self.ninputs += 1
            for dumvar in range(self.num_dummy_var):
                self.x['x_{}'.format(dumvar)] = np.zeros(self.x.shape[0])
        self.scaling = scaling
        x_scaling_mean = self.x.mean()
        y_scaling_mean = self.y.mean()
        x_scaling_std  = self.x.std()
        y_scaling_std  = self.y.std()
        self.scaling_param = {"mean": [x_scaling_mean, y_scaling_mean], "std": [x_scaling_std, y_scaling_std]}
        if scaling == "inputs":
            self.x = (self.x - x_scaling_mean)/x_scaling_std
        elif scaling == "outputs":
            self.y = (self.y - y_scaling_mean)/y_scaling_std
        elif scaling == "both":
            self.x = (self.x - x_scaling_mean)/x_scaling_std
            self.y = (self.y - y_scaling_mean)/y_scaling_std
        elif not scaling:
            self.scaling = "None"
        else:
            raise ScalingError
        
        Ts = [1] + [1.04**k for k in range(1, 20)]
        self.pms = Parallel(
            Ts,
            variables= self.x.columns.tolist(),
            parameters=['a%d' % i for i in range(self.npar)],
            x=self.x, y=self.y,
            prior_par=self.prior_par,
            ops = self.ops
        )
        
    def run_BMS(self, mcmcsteps:int = 232, save_distance:list = [], save_folder_path:str = r".\saved_model.pkl", initialize=True):
        """Runs the BMS for a number of mcmcsteps. Also saves the resultant model in a .pkl each save_distance points.

        Args:
            mcmcsteps (int, optional): Number of mcmcsteps to run. Defaults to 232.
            save_distance (int, optional): Number of mcmcsteps before saving a .pkl file. Defaults to None.
            save_folder_path (str, optional): Path to the saved .pkl file. Defaults to r".\saved_model.pkl".
        """
        self.ntrainsteps = mcmcsteps
        if initialize:
            self.description_lengths, self.mdl, self.mdl_model = [], np.inf, None
        starttime = time.time()
        for i in range(mcmcsteps):
            # MCMC update
            self.pms.mcmc_step() # MCMC step within each T
            self.pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
            # Add the description length to the trace
            self.description_lengths.append(self.pms.t1.E)
            # Check if this is the MDL expression so far
            # Thinning here
            if self.pms.t1.E < self.mdl:
                self.mdl, self.mdl_model = self.pms.t1.E, deepcopy(self.pms.t1)
            # Update the progress bar
            #f.value += 1
            #f.description = 'Run:{0}'.format(i)
            # Save pickle
            if not self.no_save:
                for save_distance_entry in save_distance:
                    if (i+1) == save_distance_entry and (i+1) != mcmcsteps:
                        intermediate_time = time.time()
                        intermediate_traintime = intermediate_time - starttime
                        self.traintime = intermediate_traintime
                        self.intermediate_trainsteps = save_distance_entry
                        with open(save_folder_path + "\\" + r'{2}_Out{3}_Scale{0}_{1}_intermediateSaveOf_{4}_totalTrainSteps.pkl'.format(self.scaling, (i+1), self.experiment, self.chosen_output, self.ntrainsteps), 'wb') as outp:
                            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
                    
    def plot_dlength(self):
        """
        Plot the description length graph, as well as information about the best found model.
        """
        ## WE check some stats:
        # print('Best model:\t', self.mdl_model)
        # print('Desc. length:\t', self.mdl)
        # print("BIC:\t", self.mdl_model.bic)
        # print("Par:\t", self.mdl_model.par_values["d0"])
        ## We display the DL
        plt.figure(figsize=(15, 5))
        plt.plot(self.description_lengths)
        
        plt.xlabel('MCMC step', fontsize=14)
        plt.ylabel('Description length', fontsize=14)
        
        plt.title('MDL model: $%s$' % self.mdl_model.latex())
        plt.show()
    
    def save_txt(self, file_path: str):
        """Saves the BMS equation in a .txt file, as well as the value of the parameters

        Args:
            file_path (str): Path to the txt file
        """
        with open(file_path, "w") as f:
            f.writelines([str(self.mdl_model), str(self.mdl_model.par_values["d0"])] )
            