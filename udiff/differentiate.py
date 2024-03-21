import random 
random.seed(0)
import numpy as np
from sympy import symbols, diff
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.linalg import solve
from typing import Tuple


class differentiator:
    """This class is used to differentiate data or functions.

    Args: 
        smoothing_obj (object, required): An object that was used to smooth the data. \
            The object should have at least the entries: (a) type_of-smooth: Defines if the smoothed function \
                is in sympy or polynomial (or other) (b) form model: Defines the smoothed model function \
                (either sympy expression or details of a polynomial) (c) x_smooth: The abscissa data of the smooth. Format: [n,]
        xdata (array, required): If no smoothing_obj is given, one can also hand over discrete abscissa data (i.e., time vector). Format: [n,]
        ydata (array, required): If no smoothing_obj is given, use this as y-data (i.e., concentration profile). Format: [n,]
    
    Stores:
        :x_diff (array): x-values of the differentiated data.
        :y_diff (array): differentiated data.
    """

    # ===================================================================================
    def __init__(   self,
                    smoothing_obj=None,
                    xdata = None,
                    ydata = None,
                    ) -> None:
        """ Initializer function.
        """
        # Check if discrete points are given to be numerically differentiated
        # Prioritise the smoothing object
        if smoothing_obj is None and ydata is None:
            raise ValueError('[-] Need either discrete data or an object with a smoothed function!')
        
        if smoothing_obj is not None:
            self.type_of_data = smoothing_obj.type_of_smooth
            self.obj = smoothing_obj
        else:
            self.type_of_data = 'discrete'
            if xdata is None:
                raise ValueError('[-] Need discrete x-data!')
            if ydata is None:
                raise ValueError('[-] Need discrete y-data!')
            self.xdata = xdata
            self.ydata = ydata
          
    # ===================================================================================
    def differentiate(  self, 
                        type_of_numerical_differentiation=None,
                        alpha=None,
                        no_opt_steps=None) -> None:
        '''Check which differentiation method should be applied. 
        
        Args:
            type_of_numerical_differentiation: In case of an absent objecet in the __init__ function \
                and the presence of discrete data, this type defines which numerical differentiation \
                method that should be used. Currently, total variation regularized differentiation (TVRD) \
                or forward finite derivatives (FD) are available as methods.
        '''

        if self.type_of_data == 'sympy':
            self.diff_sympy()
        elif self.type_of_data == 'polynomial':
            self.diff_polynomial()
        elif self.type_of_data == 'discrete':
            if type_of_numerical_differentiation == 'TVRD':
                if alpha is None or no_opt_steps is None:
                    raise ValueError('[-] Need to indicated values for >>alpha<< and >>no_opt_steps<<!')
                self.diff_numerical_TVRD(alpha, no_opt_steps)
            elif type_of_numerical_differentiation == 'FD':
                self.diff_numerical_FD()
            elif type_of_numerical_differentiation == 'BD':
                self.diff_numerical_BD()
            else:
                raise ValueError('[-] Need to define a numerical differentiation method!')
        else: 
            raise ValueError('[-] Type of data not recognized')

    # ===================================================================================
    def diff_sympy(self) -> None:
        '''Analytical differentiation of a sympy expression and evaluation of the given
        abszissa data for the derivative.
        '''
        x = symbols('x')
        sympy_expr = self.obj.model
        # Differentiate
        try:
            diff_sympy_expr = diff(sympy_expr, x)
        except:
            try:
                diff_sympy_expr = diff(sympy_expr)
            except:
                from pdb import set_trace as st; st() # Stop in debugger and check what the problem is
        xdata = self.obj.x_smooth
        y = []
        for xi in xdata:

            # First, add very small value to the datapoint if it is zero
            if xi == 0:
                eps = 1e-16
                xi += eps
            
            # First, try to evaluate the data point
            try:
                # Evaluate at x-data point
                diffvalue = diff_sympy_expr.subs(x, xi)
            except:
                # If this did not work, check problems
                from pdb import set_trace as st; st() # Stop in debugger and check what the problem is

            # Now try to convert to a float
            try:
                storevalue = float(diffvalue)
            except:
                # If convert did not work, check if value is complex, if yes, only take real part
                if isinstance(diffvalue, complex):
                    diffvalue = np.real(diffvalue)
                    storevalue = float(diffvalue)
                else:
                    # If this was not the problem, stop in a debugger and check out the problem
                    from pdb import set_trace as st; st() # Stop in debugger and check what the problem is
            
            # If everything worked, append to list
            y.append(storevalue)

        self.y_diff = np.array(y)
        self.x_diff = xdata

    # ===================================================================================
    def diff_polynomial(self) -> None:
        '''Analytical differentiation of a polynomial function and evaluation of the given
        abszissa data for the derivative.
        '''
        # Get polynomial coefficients
        polycoefs = self.obj.model['p']
        # Derive polynomial
        diff_polycoefs = np.polyder(polycoefs)
        # Evaluate polynomial at points
        xdata = self.obj.x_smooth
        y = np.polyval(diff_polycoefs, xdata)
        # Store things
        self.y_diff = y
        self.x_diff = xdata

    # ===================================================================================
    def diff_numerical_TVRD(self, alpha, no_opt_steps):
        '''Numerical differentiation by using total variation regularized differentiation (TVRD).
        
        Args: 
            alpha (int, required): Regularization parameter
            no_opt_steps (int, required): Number of optimization steps
        '''
        if alpha is None or no_opt_steps is None:
            raise ValueError('[-] Need to define a value for alpha (regularization) and the number of optimization steps!')
        obj = TVRD()
        x_diff, y_diff = obj.derive(data = self.ydata, 
                                    abszissa = self.xdata,
                                    deriv_guess = np.zeros(len(self.ydata)+1), 
                                    alpha = alpha,
                                    no_opt_steps = no_opt_steps
                                    )
        self.x_diff = x_diff
        self.y_diff = y_diff

    # ===================================================================================
    def diff_numerical_FD(self):
        '''Numerical differentiation by using forward finite derivatives (FD).
        '''
        obj = ForwardFiniteDifference(x=self.xdata, y=self.ydata)
        x_diff, y_diff = obj.derive()
        self.x_diff = x_diff
        self.y_diff = y_diff

    # ===================================================================================
    def diff_numerical_BD(self):
        '''Numerical differentiation by using backward finite derivatives (FD).
        '''
        obj = BackwardFiniteDifference(x=self.xdata, y=self.ydata)
        x_diff, y_diff = obj.derive()
        self.x_diff = x_diff
        self.y_diff = y_diff

    # ===================================================================================
    def compare_calculated_and_real_derivatives(self, calc_diff, real_diff, type_of_comp) -> None:
        '''Comparison of the calculated and real derivatives, if they are available. The coefficient
        of determination (R2-value) is anyway calculated and stored.
        
        Args: 
            calc_diff (array, required): Calculated derivatives as an array [n,]
            real_diff (array, required): Underlying ground truth of the derivatives as an array [n,]
            type_of_comp (list, required): A list of comparisons. Currently the following options \
                are available: 'OVP' (observed-vs-predicted plot) or 'profile' (the time profiles of the derivatives)
        '''

        if not isinstance(type_of_comp, list):
            raise ValueError('[-] type_of_comp needs to be a list!')
        
        # Do anyway R2 calculation
        self.R2 = r2_score(real_diff, calc_diff)
        
        if 'OVP' in type_of_comp:
            fig = plt.figure()
            ax = plt.gca()
            # Plott scatters
            plt.scatter(real_diff, 
                        calc_diff, 
                        color = ["blue"], alpha=0.5, label='', s = 8)    
            # Plot line
            cylim = ax.get_ylim()
            cxlim = ax.get_xlim()
            lims = [np.min([cxlim, cylim]),  # min of both axes
                    np.max([cxlim, cylim])]  # max of both axes
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            # Descriptions
            plt.xlabel('Observed')
            plt.ylabel('Predicted')

        if 'profile' in type_of_comp:
            fig = plt.figure()
            ax = plt.gca()
            plt.plot(range(len(real_diff)), real_diff, 'kx--', label='Real')
            plt.plot(range(len(calc_diff)), calc_diff, 'r-', label='Calculated')
            plt.xlabel('x')
            plt.ylabel('dy')
            plt.legend(frameon=False)
            
            
class TVRD:

    def __init__(self):
        """Differentiate with TVR.
        """        
        self.dy = []
        self.derivx = []

    def _make_d_mat(self) -> np.array:
        """Make differentiation matrix with central differences. NOTE: not efficient!
        Returns:
            np.array: N x N+1
        """
        arr = np.zeros((self.n,self.n+1))
        for i in range(0,self.n):
            arr[i,i] = -1.0
            arr[i,i+1] = 1.0
        return arr / self.dx

    # TODO: improve these matrix constructors
    def _make_a_mat(self) -> np.array:
        """Make integration matrix with trapezoidal rule. NOTE: not efficient!
        Returns:
            np.array: N x N+1
        """
        arr = np.zeros((self.n+1,self.n+1))
        for i in range(0,self.n+1):
            if i==0:
                continue
            for j in range(0,self.n+1):
                if j==0:
                    arr[i,j] = 0.5
                elif j<i:
                    arr[i,j] = 1.0
                elif i==j:
                    arr[i,j] = 0.5
        
        return arr[1:] * self.dx

    def _make_a_mat_t(self) -> np.array:
        """Transpose of the integration matirx with trapezoidal rule. NOTE: not efficient!
        Returns:
            np.array: N+1 x N
        """
        smat = np.ones((self.n+1,self.n))
        
        cmat = np.zeros((self.n,self.n))
        li = np.tril_indices(self.n)
        cmat[li] = 1.0

        dmat = np.diag(np.full(self.n,0.5))

        vec = np.array([np.full(self.n,0.5)])
        combmat = np.concatenate((vec, cmat - dmat))

        return (smat - combmat) * self.dx

    def make_en_mat(self, deriv_curr : np.array) -> np.array:
        """Diffusion matrix
        Args:
            deriv_curr (np.array): Current derivative of length N+1
        Returns:
            np.array: N x N
        """
        eps = pow(10,-8)
        vec = 1.0/np.sqrt(pow(self.d_mat @ deriv_curr,2) + eps)
        return np.diag(vec)

    def make_ln_mat(self, en_mat : np.array) -> np.array:
        """Diffusivity term
        Args:
            en_mat (np.array): Result from make_en_mat
        Returns:
            np.array: N+1 x N+1
        """
        return self.dx * np.transpose(self.d_mat) @ en_mat @ self.d_mat

    def make_gn_vec(self, deriv_curr : np.array, data : np.array, alpha : float, ln_mat : np.array) -> np.array:
        """Negative right hand side of linear problem
        Args:
            deriv_curr (np.array): Current derivative of size N+1
            data (np.array): Data of size N
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat
        Returns:
            np.array: Vector of length N+1
        """
        return self.a_mat_t @ self.a_mat @ deriv_curr - self.a_mat_t @ (data - data[0]) + alpha * ln_mat @ deriv_curr
    
    def make_hn_mat(self, alpha : float, ln_mat : np.array) -> np.array:
        """Matrix in linear problem
        Args:
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat
        Returns:
            np.array: N+1 x N+1
        """
        return self.a_mat_t @ self.a_mat + alpha * ln_mat
    
    def get_deriv_tvr_update(self, data : np.array, deriv_curr : np.array, alpha : float) -> np.array:
        """Get the TVR update
        Args:
            data (np.array): Data of size N
            deriv_curr (np.array): Current deriv of size N+1
            alpha (float): Regularization parameter
        Returns:
            np.array: Update vector of size N+1
        """

        n = len(data)
    
        en_mat = self.make_en_mat(
            deriv_curr=deriv_curr
            )

        ln_mat = self.make_ln_mat(
            en_mat=en_mat
            )

        hn_mat = self.make_hn_mat(
            alpha=alpha,
            ln_mat=ln_mat
            )

        gn_vec = self.make_gn_vec(
            deriv_curr=deriv_curr,
            data=data,
            alpha=alpha,
            ln_mat=ln_mat
            )

        return solve(hn_mat, -gn_vec)

    def derive(self, 
        abszissa : np.array,
        data : np.array, 
        deriv_guess : np.array, 
        alpha : float = 0.1,
        no_opt_steps : int = 1000,
        return_progress : bool = False, 
        return_interval : int = 1
        ) -> Tuple[np.array,np.array]:
        """Get derivative via TVR over optimization steps
        Args:
            abszissa (np.array): Abszissa values of the data, size N
            data (np.array): Data of size N
            deriv_guess (np.array): Guess for derivative of size N+1
            alpha (float): Regularization parameter
            no_opt_steps (int): No. opt steps to run
            return_progress (bool, optional): True to return derivative progress during optimization. Defaults to False.
            return_interval (int, optional): Interval at which to store derivative if returning. Defaults to 1.
        Returns:
            Tuple[np.array,np.array]: First is the final derivative of size N+1, second is the stored derivatives if return_progress=True of size no_opt_steps+1 x N+1, else [].
        """
        
        self.n = len(list(abszissa))
        self.dx = abszissa[1] - abszissa[0]
        
        self.dy = []
        self.derivx = []

        self.d_mat = self._make_d_mat()
        self.a_mat = self._make_a_mat()
        self.a_mat_t = self._make_a_mat_t()

        deriv_curr = deriv_guess

        if return_progress:
            deriv_st = np.full((no_opt_steps+1, len(deriv_guess)), 0)
        else:
            deriv_st = np.array([])

        for opt_step in range(0,no_opt_steps):
            update = self.get_deriv_tvr_update(
                data=data,
                deriv_curr=deriv_curr,
                alpha=alpha
                )

            deriv_curr += update

            if return_progress:
                if opt_step % return_interval == 0:
                    deriv_st[int(opt_step / return_interval)] = deriv_curr

        # adjust time to be in middle of two measured points
        shifted_time = np.array([])
        for tt in range(1,len(list(abszissa))):
            shifted_time = np.append(shifted_time, abszissa[tt-1]+((abszissa[tt]-abszissa[tt-1])/2) )

        # save variables in object    
        self.derivx = abszissa[:-1]
        self.dy = deriv_curr[1:-1]

        return self.derivx, self.dy
    
#%% 
class ForwardFiniteDifference:
    """Derivative calculation with finite forward difference. Stores derivatives as derivx (abszissa \
    of corresponding finite diff (midpoints of x)) and dy (derivative of data).
    
    Args:
        x (array, required):              abszissa (i.e., time), nparray of shape [n,] or [n,1]
        y (array, required):              data (i.e., concentration), nparray of shape [n,] or [n,1]
    """
    def __init__(self, x=None, y=None) -> None:
        assert x is not None, '[-] Need discrete x-data!'
        assert y is not None, '[-] Need discrete y-data!'
        self.x = x
        self.y = y
            
    def derive(self):
        """Derive data with forward finite difference.
        """
        x = self.x
        y = self.y
        
        if type(x) is not np.ndarray or type(y) is not np.ndarray:
            raise ValueError('[-] Inputs x and y have to be numpy arrays, each with shape [n,] or [n,1]!')
        
        self.derivx = x
        self.y = y
        dx = []
        dy = []
        for i in range(0,len(y)-1):
            diffy = y[i+1] - y[i]
            diffx = x[i+1] - x[i]
            dy.append( diffy/diffx )
            dx.append( diffx/2 + x[i] )
        self.derivx = np.array(dx)
        self.dy = np.array(dy)
        return self.derivx, self.dy
    
#%% 
class BackwardFiniteDifference:
    """Derivative calculation with finite backward difference. Stores derivatives as derivx (abszissa \
    of corresponding finite diff (midpoints of x)) and dy (derivative of data).
    
    Args:
        x (array, required):              abszissa (i.e., time), nparray of shape [n,] or [n,1]
        y (array, required):              data (i.e., concentration), nparray of shape [n,] or [n,1]
    """
    def __init__(self, x=None, y=None) -> None:
        assert x is not None, '[-] Need discrete x-data!'
        assert y is not None, '[-] Need discrete y-data!'
        self.x = x
        self.y = y
            
    def derive(self):
        """Derive data with backward finite difference.
        """
        x = self.x
        y = self.y
        
        if type(x) is not np.ndarray or type(y) is not np.ndarray:
            raise ValueError('[-] Inputs x and y have to be numpy arrays, each with shape [n,] or [n,1]!')
        
        self.derivx = x
        self.y = y
        dx = []
        dy = []
        for i in range(1,len(y)):
            diffy = y[i] - y[i-1]
            diffx = x[i] - x[i-1]
            dy.append( diffy/diffx )
            dx.append( diffx/2 + x[i-1] )
        self.derivx = np.array(dx)
        self.dy = np.array(dy)
        return self.derivx, self.dy