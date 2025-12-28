######## Imports  ####################
from pathlib import Path
import sys
from typing import List, Dict
import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from IPython.display import display
from numba import njit
from scipy.special import gamma

######################################


####### GLOBAL VARIABLES ############  
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

#file must be contained in the notebooks folder and the name of the file is spy_all.csv, which contains all the days concatenated
CSV_PATH = PROJECT_ROOT / 'notebooks' / 'spy_all.csv'

# Parameters
SUBSAMPLING_SECONDS = 5
WINDOWS = [60, 120, 150]              
N_LAGS_QV = 10                       

PRICE_TRUNCATION = 'STD3'
VOL_TRUNCATION = 'STD3'

H_MIN, H_MAX, H_MESH = 0.01, 0.499, 0.001
######################################

###### REMOVING DATES #########
FOMC = [
    "2012-01-25",
    "2012-03-13",
    "2012-04-25",
    "2012-06-20",
    "2012-08-01",
    "2012-09-13",
    "2012-10-24",
    "2012-12-12",

    "2013-01-30",
    "2013-03-20",
    "2013-05-01",
    "2013-06-19",
    "2013-07-31",
    "2013-09-18",
    "2013-10-30",
    "2013-12-18",

    "2014-01-29",
    "2014-03-19",
    "2014-04-30",
    "2014-06-18",
    "2014-07-30",
    "2014-09-17",
    "2014-10-29",
    "2014-12-17",

    "2015-01-28",
    "2015-03-18",
    "2015-04-29",
    "2015-06-17",
    "2015-07-29",
    "2015-09-17",
    "2015-10-28",
    "2015-12-16",

    "2016-01-27",
    "2016-03-16",
    "2016-04-27",
    "2016-06-15",
    "2016-07-27",
    "2016-09-21",
    "2016-11-02",
    "2016-12-14",

    "2017-02-01",
    "2017-03-15",
    "2017-05-03",
    "2017-06-14",
    "2017-07-26",
    "2017-09-20",
    "2017-11-01",
    "2017-12-13",

    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",

    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-30",
    "2019-12-11",

    "2020-01-29",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",

    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",

    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
]

TRADING_HALT = [
    '2013-07-03', 
    '2013-11-29', 
    '2013-12-24',

    '2014-07-03', 
    '2014-10-30', 
    '2014-11-28', 
    '2014-12-24', 
    
    '2015-11-27', 
    '2015-12-24', 
    
    '2016-11-25', 
    
    '2017-07-03', 
    '2017-11-24', 
    
    '2018-07-03', 
    '2018-11-23', 
    '2018-12-24',

    "2019-07-03",
    "2019-08-12",
    "2019-11-29",
    "2019-12-24",

    "2020-03-09",
    "2020-03-12",
    "2020-03-16",
    "2020-03-18",
    "2020-11-27",
    "2020-12-24",

    "2021-05-05",
    "2022-11-26",
    
    "2022-11-25"
]
######################################


######## FROM PROFESSOR's CODE ########
class Timer:
    def __init__(self, ndates, type="date"):
        self.ndates = ndates  # Number of steps/dates to process
        self.start_time = None  # Start time of the timer
        self.last_step = None   # Time of the last step
        self.type = type

    def start(self):
        """
        Initialize the timer and mark the starting point.
        """
        self.start_time = time.time()
        self.last_step = self.start_time

    def step(self, i=None):
        """
        Track progress for step `i` and display elapsed/estimated times.
        
        Args:
            i (int): Current step index (0-based).
        """
        if self.start_time is None:
            raise ValueError("Timer not started. Call start_timer() first.")
        
        now = time.time()  # Current time
        total_elapsed_time = now - self.start_time  # Total time elapsed
        step_elapsed_time = now - self.last_step  # Time since the last step

        # Update the last_step to the current time
        self.last_step = now
        
        # Calculate progress
        if i is not None:
            processed_percentage = i / self.ndates
            estimated_total_time = total_elapsed_time / processed_percentage if processed_percentage > 0 else 0
            remaining_time = estimated_total_time - total_elapsed_time
            estimated_finish = time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(self.start_time + estimated_total_time))
            
            # Print progress information
            print(f"Processing {self.type} {i+1}/{self.ndates}:\t"
                  f"Elapsed: {total_elapsed_time:.2f}s (+{step_elapsed_time:.2f}s),\t"
                  f"Remaining: {remaining_time:.2f}s,\t"
                  f"Estimated Finish: {estimated_finish}")
        else:
            print(f"Step completed. Time since last step: {step_elapsed_time:.2f}s")
    
    def total_time(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start_timer() first.")
        
        now = time.time() 
        total_elapsed_time = now - self.start_time
        return total_elapsed_time
    
class Price:
    def __init__(self, df):
        # Ensure that DT is datetime
        if 'DT' not in df.columns:
            raise ValueError("DataFrame must have a 'DT' column.")

        df['DT'] = pd.to_datetime(df['DT'], errors='coerce')
        if df['DT'].isnull().any():
            raise ValueError("Some DT values could not be converted to datetime.")

        if 'Price' not in df.columns:
            raise ValueError("DataFrame must have a 'price' column.")

        self.df = df.sort_values('DT').reset_index(drop=True)  # Ensure sorted by DT

        self.delta = 1.0 / (252.0 * 23400.0)

    def TRADING_HALT(self, duration=300):
        """
        Check for trading halts. A trading halt is identified if the gap between 
        consecutive timestamps is greater than 'duration' seconds.

        Returns:
            A Pandas Series of time gaps (in seconds) that exceed the duration.
            If empty, no halts are present.
        """
        time_diffs = self.df['DT'].diff().dt.total_seconds().fillna(0)
        halts = time_diffs[time_diffs > duration]
        return halts

    def subsample(self, sub=5):
        """
        Subsample the data every 'sub' seconds. The method sets 'DT' as index,
        resamples the data at 'sub' second intervals, taking the last 
        observed price in each interval.
        """
        self.df = self.df[::sub]
        self.delta = sub * self.delta

    def get_price(self):
        """
        Returns:
            Numpy array of the prices.
        """
        return self.df['Price'].values
    
    def get_DT(self):
        """
        Returns:
            Numpy array of the datetimes.
        """
        return self.df['DT'].values

    def get_increments(self):
        """
        Returns:
            Numpy array of the increments (differences) of consecutive prices.
        """
        increments = self.df['price'].diff().dropna().values
        return increments



def Phi_Hl(l: int, H: float) -> float:
    """
    Compute the value of $\\Phi^H_\\ell$ using a finite difference formula.

    This function evaluates a discrete approximation based on powers of absolute values,
    commonly used in fractional Brownian motion and related models.

    :param l: Index $\\ell$ in the formula (integer).
    :param H: Hurst exponent $H$, controlling the memory effect (float).
    :return: Computed value of $\\Phi^H_\\ell$.
    """
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    return numerator / denominator




def dPhi_Hl_dH(l: int, H: float) -> float:
    """
    Compute the derivative of $\\Phi^H_\\ell$ with respect to $H$.

    Uses the chain rule to differentiate power terms in the finite difference formula.

    :param l: Index $\\ell$ in the formula (integer).
    :param H: Hurst exponent $H$ (float).
    :return: The computed derivative $\\frac{d}{dH} \\Phi^H_\\ell$.
    """
    def power_term_derivative(x, H):
        if x == 0:
            return 0
        return (2 * x ** (2 * H + 2) * np.log(np.abs(x)))
    
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))

    numerator_derivative = (
        power_term_derivative(np.abs(l + 2), H) - 4 * power_term_derivative(np.abs(l + 1), H) +
        6 * power_term_derivative(np.abs(l), H) - 4 * power_term_derivative(np.abs(l - 1), H) +
        power_term_derivative(np.abs(l - 2), H)
    )
    
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    denominator_derivative = 4 * (4 * H + 3)
    
    return (numerator_derivative * denominator - denominator_derivative * numerator) / (denominator * denominator)


def dichotomic_search(f, target: float, low: float, high: float, is_increasing: bool = True, epsilon: float = 1e-5) -> float:
    """
    Perform a dichotomic (binary) search to find an approximate solution $x$ such that $f(x) \\approx \\text{target}$.

    The function `f` is assumed to be monotonic (either increasing or decreasing, default being increasing).
    
    :param f: Monotonic function to search over.
    :param target: Target value to search for.
    :param low: Lower bound of the search interval.
    :param high: Upper bound of the search interval.
    :param is_increasing: If True, `f` is increasing; otherwise, it is decreasing.
    :param epsilon: Tolerance for stopping the search.
    :return: Approximate solution $x$ where $f(x) \\approx \\text{target}$.
    """

    if is_increasing:
        if f(low) > target:
            return low
        if f(high) < target:
            return high
    else:
        if f(low) < target:
            return low
        if f(high) > target:
            return high

    while low <= high:
        mid = (low + high) / 2
        f_mid = f(mid)

        if abs(f_mid - target) < epsilon:
            return mid  # Found a value close to the target

        if is_increasing:
            if f_mid < target:
                low = mid + epsilon
            else:
                high = mid - epsilon
        else:
            if f_mid > target:
                low = mid + epsilon
            else:
                high = mid - epsilon

    return None  # Target value not found within the interval

def F_estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H: list, normalisation: float = 1) -> float:
    """
    Compute the GMM objective function $F(H, R)$ for given parameters.
    
    This function minimizes:
    
    $$ F(H, R) = (V - P)^T W (V - P) $$
    
    where $P$ is computed based on $H$.

    :param W: Weight matrix (numpy array).
    :param V: Observation vector (numpy array).
    :param Psi_func: Function $\\Psi(H)$ providing model predictions.
    :param H: Scalar Hurst exponent wrapped in a list.
    :param normalisation: Normalization factor for the function value.
    :return: Evaluated objective function value.
    """

    H = H[0]
    V = np.atleast_2d(V).reshape(-1, 1)
    Psi = np.atleast_2d(Psi_func(H)).reshape(-1, 1)
        
    term0 = V.T @ W @ V
    term1 = (Psi.T @ W @ V) + (V.T @ W @ Psi)
    term2 = Psi.T @ W @ Psi
    
    term0 = term0[0, 0]
    term1 = term1[0, 0]
    term2 = term2[0, 0]
    
    R = term1 / term2 / 2
    
    return normalisation * (term0 - R * term1 + term2 * R * R)

def F_GMM_get_R(W: np.ndarray, V: np.ndarray, Psi_func, H: float) -> float:
    V = np.atleast_2d(V).reshape(-1, 1)
    Psi = np.atleast_2d(Psi_func(H)).reshape(-1, 1)
        
    term0 = V.T @ W @ V
    term1 = (Psi.T @ W @ V) + (V.T @ W @ Psi)
    term2 = Psi.T @ W @ Psi
    
    term0 = term0[0, 0]
    term1 = term1[0, 0]
    term2 = term2[0, 0]
    
    R = term1 / term2 / 2
    
    return R

def estimation_GMM(W: np.ndarray, V: np.ndarray, Psi_func, H_min: float = 0.001, H_max: float = 0.499, mesh: float = 0.001, debug: bool = False):
    """
    Perform Generalized Method of Moments (GMM) estimation for the Hurst exponent.
    
    This method finds $H$ that minimizes the GMM objective function over a predefined grid.
    
    :param W: Weight matrix (numpy array).
    :param V: Observation vector (numpy array).
    :param Psi_func: Function returning model predictions $\\Psi(H)$.
    :param H_min: Minimum value for H search grid.
    :param H_max: Maximum value for H search grid.
    :param mesh: Step size for grid search.
    :param debug: If True, return intermediate results.
    :return: Estimated Hurst exponent.
    """
    H_values = np.arange(H_min, H_max, mesh)
    F_values = [F_estimation_GMM(W, V, Psi_func, [H]) for H in H_values]
    min_index = np.argmin(F_values)
    
    if debug:
        R_values = [F_GMM_get_R(W, V, Psi_func, H) for H in H_values]
        return H_values, F_values, min_index, R_values

    return H_values[min_index], F_GMM_get_R(W, V, Psi_func, H_values[min_index])

def uncorrected_alpha(theta, lag, H):
    return theta**(2*H-1) * dPhi_Hl_dH(lag, H) + 2 * np.log(theta) * Phi_Hl(lag, H)

def uncorrected_beta(theta, lag, H):
    return theta**(2*H-1) * Phi_Hl(lag, H)

def variation_44(f):
    shifts = [-2, -1, 0, 1, 2]
    coefficients = [1, -4, 6, -4, 1]
    return sum([
        c1 * c2 * f(s1, s2)
        for (s1, c1) in zip(shifts, coefficients)
        for (s2, c2) in zip(shifts, coefficients)
    ])
        
def uncorrected_gamma(theta1, theta2, lag1, lag2, H):
    if H == 0.25:
        local_f = lambda l1, l2: np.abs((l1 + lag1) * theta1 + (l2 + lag2) * theta2)**(6) * np.log(np.abs((l1 + lag1) * theta1 + (l2 + lag2) * theta2)) + np.abs((l1 + lag1) * theta1 - (l2 + lag2) * theta2)**(6) * np.log(np.abs((l1 + lag1) * theta1 - (l2 + lag2) * theta2))
        return variation_44(local_f) / ( 5760 * (theta1 * theta2)**3)
    else:
        local_f = lambda l1, l2: np.abs((l1 + lag1) * theta1 + (l2 + lag2) * theta2)**(4 * H + 5) + np.abs((l1 + lag1) * theta1 - (l2 + lag2) * theta2)**(4 * H + 5)
        return gamma(1+2*H)**2 * (1-1/np.cos(2*np.pi * H)) * variation_44(local_f) / ( 4 * gamma(6+4*H) * (theta1 * theta2)**(2*H + 5))

def compute_alpha(theta, lag, H):
    if lag == 1:
        return uncorrected_alpha(theta, 0, H) + 2 * uncorrected_alpha(theta, 1, H)
    return uncorrected_alpha(theta, lag, H)

def compute_beta(theta, lag, H):
    if lag == 1:
        return uncorrected_beta(theta, 0, H) + 2 * uncorrected_beta(theta, 1, H)
    return uncorrected_beta(theta, lag, H)

def compute_gamma(theta1, theta2, lag1, lag2, H):
    if lag1 == 1 and lag2 == 1:
        return uncorrected_gamma(theta1, theta2, 0, 0, H) + 2 * uncorrected_gamma(theta1, theta2, 0, 1, H) + 2 * uncorrected_gamma(theta1, theta2, 1, 0, H) + 4 * uncorrected_gamma(theta1, theta2, 1, 1, H)
    elif lag1 == 1 and lag2 > 1:
        return uncorrected_gamma(theta1, theta2, 0, lag2, H) + 2 * uncorrected_gamma(theta1, theta2, 1, lag2, H) 
    elif lag1 > 1 and lag2 == 1:
        return uncorrected_gamma(theta1, theta2, lag1, 0, H) + 2 * uncorrected_gamma(theta1, theta2, lag1, 1, H) 
    else:
        return uncorrected_gamma(theta1, theta2, lag1, lag2, H)
    
def get_optimal_variance(params_volatility, H, eta, t, delta_n):
    R_t = eta ** 2 * t
    Gamma_t = eta ** 4 * t
    
    window_values = []
    lags_values = []

    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']
        
        window_values.extend([window for _ in range(N_lags)])
        lags_values.extend([i for i in range(1,N_lags+1)])

    reference_window = window_values[0]

    theta = [w / reference_window for w in window_values]

    m = len(window_values)

    alpha = np.zeros(m)
    beta = np.zeros(m)

    for i in range(m):
        alpha[i] = compute_alpha(theta[i], lags_values[i], H)
        beta[i] = compute_beta(theta[i], lags_values[i], H)


    Sigma = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            Sigma[i,j] = compute_gamma(theta[i], theta[j], lags_values[i], lags_values[j], H) * (theta[i] * theta[j])**(2*H-1/2)

    u_t = np.array([alpha * R_t, beta]).transpose()

    D = np.array([
        [1, 0],
        [-2 * np.log(reference_window * delta_n) * R_t, 1]
    ])

    Sigma = Sigma * Gamma_t

    uSinvu_inv = np.linalg.inv(u_t.transpose() @ np.linalg.inv(Sigma) @ u_t)
    matrix_43 = reference_window * delta_n * D @ uSinvu_inv @ D.transpose()

    return matrix_43[0,0]**0.5, matrix_43[1,1]**0.5

def get_theoretical_variance(params_volatility, H):
    window_values = []
    lags_values = []

    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']
        
        window_values.extend([window for _ in range(N_lags)])
        lags_values.extend([i for i in range(1,N_lags+1)])

    reference_window = window_values[0]
    theta = [w / reference_window for w in window_values]

    m = len(window_values)

    alpha = np.zeros(m)
    beta = np.zeros(m)

    for i in range(m):
        alpha[i] = compute_alpha(theta[i], lags_values[i], H)
        beta[i] = compute_beta(theta[i], lags_values[i], H)

    Sigma = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            Sigma[i,j] = compute_gamma(theta[i], theta[j], lags_values[i], lags_values[j], H) * (theta[i] * theta[j])**(2*H-1/2)
    
    return Sigma

def get_confidence_size(params_volatility, H_estimated, R_estimated, n_days, delta_n, Sigma_estimated, W_chosen):
    window_values = []
    lags_values = []

    for param in params_volatility:
        window = param['window']
        N_lags = param['N_lags']
        
        window_values.extend([window for _ in range(N_lags)])
        lags_values.extend([i for i in range(1,N_lags+1)])

    reference_window = window_values[0]
    theta = [w / reference_window for w in window_values]

    m = len(window_values)


    alpha = np.zeros(m)
    beta = np.zeros(m)

    for i in range(m):
        alpha[i] = compute_alpha(theta[i], lags_values[i], H_estimated)
        beta[i] = compute_beta(theta[i], lags_values[i], H_estimated)

    alpha_beta = np.array([alpha, beta])

    u_t = np.array([alpha * R_estimated, beta]).transpose()

    D = np.array([
        [1, 0],
        [-2 * np.log(reference_window * delta_n), 1]
    ])

    uWu_inv = np.linalg.inv(u_t.transpose() @ W_chosen @ u_t)
    matrix_43 = (delta_n * reference_window)**(1-4*H_estimated) * reference_window * delta_n * D @ uWu_inv @ u_t.transpose() @ W_chosen @ Sigma_estimated @ W_chosen @ u_t @ uWu_inv @ D.transpose()
    # matrix_43 = reference_window * delta_n * D @ uWu_inv @ u_t.transpose() @ W_chosen @ Sigma_estimated @ W_chosen @ u_t @ uWu_inv @ D.transpose()

    return matrix_43[0,0]**0.5 / np.sqrt(n_days), matrix_43[1,1]**0.5  / np.sqrt(n_days)

def bipower_average_V(price, window, delta):
    n = len(price)
    if n <= 2 * window:  # Ensure there's enough data
        print("Not enough data points.")
        return -1.0

    # Compute price increments over the given window
    price_increments = price[window:] - price[:-window]

    # Calculate bipower average volatility
    sum_ = np.sum(np.abs(price_increments[window:] * price_increments[:-window]))

    # Calculate the final result
    mean = sum_ / (n - 2 * window)
    return (mean / (delta * window)) * (np.pi / 2)

class Volatility:
    def __init__(self, values):
        """
        values: array-like of computed volatility values
        """
        self.values = np.array(values)
        
    def get_values(self):
        return self.values
        
    def rv(self, delta):
        """
        Realized variance as sum of values * delta.
        """
        return np.sum(self.values) * delta

class VolatilityEstimator:
    def __init__(self, delta, window, price_truncation="INFINITE"):
        # Ensure that the truncation method provided is one of the allowed types
        if price_truncation not in ["INFINITE", "STD3", "BIVAR3", "STD5", "BIVAR5"]:
            raise ValueError("Invalid truncation method. Choose one of: 'INFINITE', 'STD3', 'BIVAR3', 'STD5', 'BIVAR5'")

        # Store the parameters as instance variables
        self.delta = delta
        self.window = window
        self.price_truncation = price_truncation
        
    def compute(self, price):
        price = np.log(price)
        priceinc = price[1:] - price[:-1]

        truncationValue = np.inf
        if self.price_truncation == 'STD3':
            truncationValue = 3 * np.std(priceinc)
        elif self.price_truncation == 'STD5':
            truncationValue = 5 * np.std(priceinc)
        elif self.price_truncation == 'BIVAR3':
            bav = bipower_average_V(price, self.window, self.delta)
            truncationValue = 3 * np.sqrt(bav * self.delta)
        elif self.price_truncation == 'BIVAR5':
            bav = bipower_average_V(price, self.window, self.delta)
            truncationValue = 5 * np.sqrt(bav * self.delta)

        priceinc[np.abs(priceinc) > truncationValue] = 0

        # Realized variance cumulative sum
        rv = np.concatenate([[0], np.cumsum(priceinc**2)])

        # Average volatility (not necessarily needed for the returned Volatility)
        avgVol = rv[-1] / (self.delta * (len(price) - 1))
        
        # Windowed volatility estimate
        volatilities = (rv[self.window:] - rv[:-self.window]) / (self.delta * self.window)

        return Volatility(volatilities)

class VolatilityPattern:
    def __init__(self):
        self.current_pattern = None
        self.N_elements = 0

    def accumulate(self, vol):
        if isinstance(vol, list):
            # vol is a list of Volatility objects
            all_values = [v.get_values() for v in vol]
            # Normalize each by its mean
            all_values = [arr / np.mean(arr) for arr in all_values]
            # Sum across all given volatilities
            sum_values = np.sum(all_values, axis=0)

            # Add current pattern if it exists
            if self.current_pattern is not None:
                current_vals = self.current_pattern.get_values()
                min_len = min(len(sum_values), len(current_vals))
                sum_values = sum_values[:min_len] + current_vals[:min_len]

            # Update current pattern
            self.current_pattern = Volatility(sum_values)
            self.N_elements += len(vol)
        else:
            if isinstance(vol, Volatility):
                # vol is a single Volatility object
                vol_values = vol.get_values()
            else:
                vol_values = vol
            vol_values = vol_values / np.mean(vol_values)

            # If we have a current pattern, add it
            if self.current_pattern is not None:
                current_vals = self.current_pattern.get_values()
                # Align to the shortest length to avoid shape mismatches across days
                min_len = min(len(vol_values), len(current_vals))
                if min_len == 0:
                    return
                sum_values = vol_values[:min_len] + current_vals[:min_len]
            else:
                sum_values = vol_values

            self.current_pattern = Volatility(sum_values)
            self.N_elements += 1

    def get_pattern(self):
        if self.current_pattern is None or self.N_elements == 0:
            # No pattern accumulated yet
            return None
        # Return the averaged pattern
        return Volatility(self.current_pattern.get_values() / self.N_elements)
            
def volatility_pattern(vols):
    """
    vols: a list of Volatility objects with the same DT.
    
    Create a new Volatility that averages all the vols.values and then
    normalizes them so that their average value is 1.
    """
    if not vols:
        raise ValueError("No Volatility objects provided.")

    # Extract values from each
    all_values = [v.get_values() for v in vols]
    all_values = [v / np.mean(v) for v in all_values]

    # Compute average across all vols
    avg_values = np.mean(all_values, axis=0)
    # Normalize so that average is 1
    avg_val_mean = np.mean(avg_values)
    if avg_val_mean != 0:
        avg_values = avg_values / avg_val_mean

    return Volatility(avg_values)
####################################################

############# WORKFLOW #############################

def load_spy_data(csv_path: Path) -> pd.DataFrame:
    """
    param:
        csv_path: Path to the SPY tick CSV file.
    Returns:
        A pandas DataFrame with columns DT, Price, and date,
        filtered to exclude FOMC and trading-halt dates.
    """
    df_pl = pl.read_csv(csv_path)
    # Normalize column names
    cols_upper = {c.upper(): c for c in df_pl.columns}
    dt_col = next((cols_upper[key] for key in ['DT', 'DATETIME', 'TIMESTAMP', 'TIME'] if key in cols_upper), None)
    price_col = next((c for c in df_pl.columns if c.lower() in ['price', 'close', 'mid', 'px_last']), None)
    if dt_col is None or price_col is None:
        raise ValueError('Could not find DT/Price columns in CSV.')
    df_pl = df_pl.select([pl.col(dt_col).alias('DT'), pl.col(price_col).alias('Price')])
    # Convert to pandas for downstream code
    df = df_pl.to_pandas()
    df['DT'] = pd.to_datetime(df['DT'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['DT', 'Price']).sort_values('DT').reset_index(drop=True)
    df['date'] = df['DT'].dt.date
    excluded_dates = {pd.to_datetime(d).date() for d in (FOMC + TRADING_HALT)}
    df = df[~df['date'].isin(excluded_dates)].reset_index(drop=True)
    return df

@njit #decorator for HPC
def quadratic_covariations_njit(vol_increments: np.ndarray, window: int, n_lags: int) -> np.ndarray:
    """
    param:
        vol_increments: 1D array of volatility increments
        window: Number of steps per window (K)
        n_lags: Number of lags to compute (including lag 0 internally)
    Returns:
        1D array of length n_lags-1 containing covariations for lags 1..n_lags-1
    """
    cov = np.zeros(n_lags)
    n = len(vol_increments)
    for lag in range(n_lags):
        if lag == 0:
            cov[lag] = np.mean(vol_increments * vol_increments)
        else:
            m = n - lag * window
            if m <= 0:
                cov[lag] = np.nan
                continue
            s = 0.0
            for i in range(m):
                s += vol_increments[i + lag * window] * vol_increments[i]
            cov[lag] = s / m
    if n_lags > 1:
        cov[1] = cov[0] + 2 * cov[1]
    return cov[1:]


def compute_daily_volatilities(df: pd.DataFrame, window_steps: int, price_truncation: str = PRICE_TRUNCATION):
    """
    params:
        df: DataFrame with DT, Price, and date columns
        window_steps: Window size (K) in subsampling steps.
        price_truncation: Truncation method for price increments.
    Returns:
        A tuple of:
        1) daily: list of dicts with keys "date" and "vol" (Volatility objects).
        2) pattern_values: ndarray of intraday pattern values or None
        3) errors: list of error strings for days that failed processing.
    """
    pattern_builder = VolatilityPattern()
    daily = []
    errors: List[str] = []
    dates = sorted(df['date'].unique())
    timer = Timer(len(dates), type='day')
    timer.start()
    for i, date in enumerate(dates):
        timer.step(i)
        day_df = df[df['date'] == date][['DT', 'Price']].copy()
        if len(day_df) <= window_steps:
            errors.append(f"{date}: not enough points for window {window_steps}.")
            continue
        try:
            price = Price(day_df)
            price.subsample(SUBSAMPLING_SECONDS)
            ve = VolatilityEstimator(delta=price.delta, window=window_steps, price_truncation=price_truncation)
            vol = ve.compute(price.get_price())
        except Exception as exc:
            errors.append(f"{date}: {exc}")
            continue
        pattern_builder.accumulate(vol)
        daily.append({'date': date, 'vol': vol})
    pattern = pattern_builder.get_pattern()
    pattern_values = pattern.get_values() if pattern is not None else None
    return daily, pattern_values, errors


def compute_quadratic_covariations(daily, pattern_values: np.ndarray, window_steps: int, n_lags_qv: int, vol_truncation=VOL_TRUNCATION):
    """
    Steps:
    - Normalize each day's volatility by the intraday pattern and mean level.
    - Compute windowed increments and apply truncation to outliers
    - Compute quadratic covariations per day and average across days
    params:
        daily: list of dicts from compute_daily_volatilities.
        pattern_values: ndarray of intraday pattern values.
        window_steps: Window size (K) in subsampling steps.
        n_lags_qv: Number of lags (including lag 0 internally).
        vol_truncation: Truncation rule or numeric threshold.
    Returns:
        A tuple of (V_avg, V_matrix, lags) where:
        - V_avg is the average covariation across days.
        - V_matrix is the per-day covariation matrix.
        - lags is the lag index array (1..n_lags_qv-1).
    """
    if pattern_values is None:
        raise RuntimeError('No intraday volatility pattern available (not enough valid days).')
    V_by_day = []
    for entry in daily:
        vol_values = entry['vol'].get_values()
        if len(vol_values) <= window_steps:
            continue
        mean_vol = np.mean(vol_values)
        norm_vol = vol_values / pattern_values / mean_vol
        vol_inc = norm_vol[window_steps:] - norm_vol[:-window_steps]
        if isinstance(vol_truncation, str):
            if vol_truncation == 'STD3':
                trunc = 3 * np.std(vol_inc)
            elif vol_truncation == 'STD5':
                trunc = 5 * np.std(vol_inc)
            else:
                trunc = np.inf
        else:
            trunc = float(vol_truncation)
        vol_inc[np.abs(vol_inc) > trunc] = 0
        V_day = quadratic_covariations_njit(vol_inc, window_steps, n_lags_qv)
        V_by_day.append(V_day)
    if not V_by_day:
        raise RuntimeError('No quadratic covariations computed (no valid days after filtering).')
    V_matrix = np.vstack(V_by_day)
    V_avg = np.nanmean(V_matrix, axis=0)
    lags = np.arange(1, n_lags_qv)
    return V_avg, V_matrix, lags

def build_Psi_function(lags: np.ndarray, window_steps: int):
    """
    params:
        lags: 1D array of lag indices (1..n_lags-1).
        window_steps: Window size (K) in subsampling steps.
    Returns:
        A callable Psi(H) that produces model-implied covariations for a given H.
    """
    def Psi(H):
        factor = (window_steps ** (2 * H))
        out = []
        for lag in lags:
            if lag == 1:
                out.append(factor * (Phi_Hl(0, H) + 2 * Phi_Hl(1, H)))
            else:
                out.append(factor * Phi_Hl(lag, H))
        return np.array(out)
    return Psi

def estimate_H(V_avg: np.ndarray, lags: np.ndarray, window_steps: int):
    W = np.eye(len(lags))
    Psi = build_Psi_function(lags, window_steps)
    return estimation_GMM(W, V_avg, Psi, H_min=H_MIN, H_max=H_MAX, mesh=H_MESH)
run_start = time.time()
df = load_spy_data(CSV_PATH)
df.head()

# Plot SPY price series
plt.figure(figsize=(10, 4))
plt.plot(df["DT"], df["Price"], linewidth=0.7)
plt.title("SPY price series")
plt.xlabel("Time")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

results: Dict[int, Dict[str, object]] = {}
for window in WINDOWS:
    daily_results, pattern_values, errors = compute_daily_volatilities(df, window_steps=window)
    print(f"K={window}: valid days = {len(daily_results)}, errors = {len(errors)}")
    if errors:
        print('  sample errors:', errors[:3])
    V_avg, V_matrix, lags = compute_quadratic_covariations(daily_results, pattern_values, window_steps=window, n_lags_qv=N_LAGS_QV)
    H_hat, R_hat = estimate_H(V_avg, lags, window_steps=window)
    results[window] = {
        'daily': daily_results,
        'pattern': pattern_values,
        'V_avg': V_avg,
        'V_matrix': V_matrix,
        'lags': lags,
        'H_hat': H_hat,
        'R_hat': R_hat,
    }
    display(pd.DataFrame({'lag': lags, 'average_V_l_n': V_avg}))
    print(f"H = {H_hat:.4f}, R = {R_hat:.4f}")

print(f"Total runtime: {time.time() - run_start:.2f}s")

######### Plot section ####################
pattern_window = 120 if 120 in results else WINDOWS[0]
pattern_vals = results[pattern_window]['pattern']
if pattern_vals is None:
    raise RuntimeError('No pattern available to plot.')
plt.figure(figsize=(10,4))
plt.plot(pattern_vals, label=f'Pattern (K={pattern_window})')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
plt.title('Volatility pattern correction')
plt.xlabel('Window index')
plt.ylabel('Normalized volatility')
plt.legend()
pattern_path = PROJECT_ROOT / 'volatility_pattern_correction.png'
plt.tight_layout()
plt.savefig(pattern_path, dpi=150)
plt.show()
print('Saved', pattern_path)

###### Checking Results ###########
for window in WINDOWS:
    if window not in results:
        continue
    data = results[window]
    V_avg = data['V_avg']
    lags = data['lags']
    H_hat = data['H_hat']
    Psi_func = build_Psi_function(lags, window)
    Psi_vals = Psi_func(H_hat)
    plt.figure(figsize=(8,4))
    plt.plot(lags, V_avg, 'o-', label='V_avg')
    plt.plot(lags, Psi_vals, 's--', label=f'Psi(H={H_hat:.3f})')
    plt.title(f'Diagnostic: V vs Psi (K={window})')
    plt.xlabel('lag')
    plt.ylabel('value')
    plt.legend()
    diag_path = PROJECT_ROOT / f'diagnostic_V_vs_Psi_K{window}.png'
    plt.tight_layout()
    plt.savefig(diag_path, dpi=150)
    plt.show()
    print('Saved', diag_path)


# Normalized V vs Psi
for window, data in results.items():
    V = np.asarray(data['V_avg'], dtype=float)
    lags = np.asarray(data['lags'], dtype=int)
    H_hat = float(data['H_hat'])
    Psi = build_Psi_function(lags, window)(H_hat).flatten()
    if len(V) == 0 or V[0] == 0 or Psi[0] == 0:
        print(f'Skip window {window}: zero first element for normalization.')
        continue
    V_norm = V / V[0]
    Psi_norm = Psi / Psi[0]
    plt.figure(figsize=(6,3))
    plt.plot(lags, V_norm, 'o-', label='V_norm')
    plt.plot(lags, Psi_norm, 's--', label=f'Psi_norm (H={H_hat:.3f})')
    plt.title(f'Normalized V vs Psi (window K={window})')
    plt.xlabel('lag')
    plt.ylabel('normalized value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
