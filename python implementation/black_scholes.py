import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Dict,List,Union,Tuple
from scipy.stats import norm

class BlackScholes:
    """
    A class for calculating the price of an option using the black-scholes formula.
    Implement vectorized operation for efficient pricing of multiple options.
    """

    def __init__(self):
        """initialize the Black-Scholes model"""
        pass

    @staticmethod
    def _calculate_d1(S:np.ndarray, K:np.ndarray, r:float, sigma:float,T:float)->np.ndarray:
        """
            Calculate d1 parameter for the black-scholes formuala.

            Parameters:
                S (np.ndarray): Current stock price
                K (np.ndarray): Strike price
                r (float): Risk free rate
                sigma (float): Volatility
                T (float): Time to expiration

            Returns:
                d1 (np.ndarray): d1 parameter
        """
        return  (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

    @staticmethod
    def _calculate_d2(d1:np.ndarray, sigma:float,T:float)->np.ndarray:
        """
            Calculate d2 parameter for the black-scholes formuala.

            Parameters:
                d1 (np.ndarray): d1 parameter
                sigma (float): Volatility
                T (float): Time to expiration

            Returns:
                d2 (np.ndarray): d2 parameter
        """
        return d1 - sigma*np.sqrt(T)


    def call_price(self,S:Union[float,np.ndarray],K:Union[float,np.ndarray],
                    r:float,sigma:float,T:float)->np.ndarray:
        """
            Calculate the call price using the black-scholes formula.

            Parameters:
                S (Union[float,np.ndarray]): Current stock price
                K (Union[float,np.ndarray]): Strike price
                r (float): Risk free rate
                sigma (float): Volatility
                T (float): Time to expiration

            Returns:
                price (np.ndarray): Call price
        """
        #convert inputs to numpy arrays for vectorized operation
        S = np.asarray(S)
        K = np.asarray(K)

        #Handle edge cases
        if np.any(sigma<=0) or T <= 0:
            raise ValueError("Volatility and time to maturity must be positive")
        
        d1 = self._calculate_d1(S,K,r,sigma,T)
        d2 = self._calculate_d2(d1,sigma,T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    def put_price(self,S:Union[float,np.ndarray],K:Union[float,np.ndarray],
                    r:float,sigma:float,T:float)->np.ndarray:
        """
            Calculate the put price using the black-scholes formula.

            Parameters:
                S (Union[float,np.ndarray]): Current stock price
                K (Union[float,np.ndarray]): Strike price
                r (float): Risk free rate
                sigma (float): Volatility
                T (float): Time to expiration

            Returns:
                price (np.ndarray): Put price
        """
        #convert inputs to numpy arrays for vectorized operation
        S = np.asarray(S)
        K = np.asarray(K)

        #Handle edge cases
        if np.any(sigma<=0) or T <= 0:
            raise ValueError("Volatility and time to maturity must be positive")
        
        d1 = self._calculate_d1(S,K,r,sigma,T)
        d2 = self._calculate_d2(d1,sigma,T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def call_greeks(self,S:float,K:float,r:float,sigma:float,T:float)->Dict[str,float]:
        """
            Calculate greeks for a European call option.

            Parameters:
                S (float): Current stock price
                K (float): Strike price
                r (float): Risk free rate
                sigma (float): Volatility
                T (float): Time to expiration

            Returns:
                greeks (Dict[str,float]): Dictionary of greeks
        """

        if sigma <= 0 or T <= 0:
            raise ValueError("Volatility and time to maturity must be positive")
        
        d1 = self._calculate_d1(S,K,r,sigma,T)
        d2 = self._calculate_d2(d1,sigma,T)

        #calculate Greeks
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        theta = -S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
        rho = K*T*np.exp(-r*T)*norm.cdf(d2)
        vega = S*np.sqrt(T)*norm.pdf(d1)

        return {
            "delta":delta,
            "gamma":gamma,
            "theta":theta,
            "vega":vega,
            "rho":rho
        }

    def put_greeks(self,S:float,K:float,r:float,sigma:float,T:float)->Dict[str,float]:
        """
            Calculate greeks for a European put option.

            Parameters:
                S (float): Current stock price
                K (float): Strike price
                r (float): Risk free rate
                sigma (float): Volatility
                T (float): Time to expiration

            Returns:
                greeks (Dict[str,float]): Dictionary of greeks
        """

        if sigma <= 0 or T <= 0:
            raise ValueError("Volatility and time to maturity must be positive")    
        
        d1 = self._calculate_d1(S,K,r,sigma,T)
        d2 = self._calculate_d2(d1,sigma,T)

        #calculate greeks

        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        theta = -S * sigma * norm.pdf(d1)/(2*np.sqrt(T)) + r * K *np.exp(-r*T) * norm.cdf(-d2)
        vega = S*np.sqrt(T)*norm.pdf(d1)
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)

        return {
            "delta":delta,
            "gamma":gamma,
            "theta":theta,
            "vega":vega,
            "rho":rho
        }

    def implied_volatility_call(self,market_price:float,S:float,K:float,r:float,T:float
                                ,initial_guess:float=0.2,max_iteration:int=100,precision:float=1e-5)->float:

        """
            Calculate implied volatility for a European call option.

            Parameters:
                market_price (float): Market price of the option
                S (float): Current stock price
                K (float): Strike price
                r (float): Risk free rate
                T (float): Time to expiration
                initial_guess (float): Initial guess of the implied volatility
                max_iteration (int): Maximum number of iterations
                precision (float): Precision of the implied volatility
            Returns:
                implied_volatility (float): Implied volatility
        """

        sigma = initial_guess

        for i in range(max_iteration):
            price = self.call_price(S,K,r,sigma,T)
            diff = market_price - price 

            if abs(diff) < precision:
                return sigma 
            
            #calculate vega
            d1 = self._calculate_d1(S,K,r,sigma,T)
            vega = S * np.sqrt(T) * norm.pdf(d1)

            #avoid division by zero
            if abs(vega) < 1e-10:
                return np.nan 
            
            #update sigma using newton-raphson
            sigma = sigma + diff/vega 

            #ensure sigma stays positive
            if sigma <= 0:
                sigma = 0.001

        #did not converge
        return np.nan 


    def implied_volatility_put(self,market_price:float,S:float,K:float,r:float,T:float
                                ,initial_guess:float=0.2,max_iteration:int=100,precision:float=1e-5)->float:

        """
        Calculate the implied volatility of a put option.
        
        Parameters:
            market_price (float): Market price of the option
            S (float): Current stock price
            K (float): Strike price
            r (float): Risk free rate
            T (float): Time to expiration
            initial_guess (float): Initial guess of the implied volatility
            max_iteration (int): Maximum number of iterations
            precision (float): Precision of the implied volatility
        Returns:
            implied_volatility (float): Implied volatility
        """

        sigma = initial_guess 

        for i in range(max_iteration):
            price = self.put_price(S,K,r,sigma,T)
            diff = market_price - price 

            if abs(diff) < precision:
                return sigma

            #calculate vega 
            d1 = self._calculate_d1(S,K,r,sigma,T)
            vega = S * np.sqrt(T) * norm.pdf(d1)

            #calculate vega 
            d1 = self._calculate_d1(S,K,r,sigma,T)
            vega = S * np.sqrt(T) * norm.pdf(d1)

            #avoid division by zero
            if abs(vega)  < 1e-10:
                return np.nan

            #update sigma using newton-raphson
            sigma = sigma + diff/vega 

            #ensure sigma stays positive
            if sigma <= 0:
                sigma = 0.001

        #did not converge
        return np.nan
    
    def plot_option_prices(self, S_range: np.ndarray, K: float, r: float, 
                          sigma: float, T: float, option_type: str = 'both') -> None:
        """
        Plot option prices for a range of stock prices.
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of stock prices
        K : float
            Strike price
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to maturity in years
        option_type : str, optional
            Type of option to plot ('call', 'put', or 'both'), default is 'both'
        """
        plt.figure(figsize=(10, 6))
        
        if option_type in ['call', 'both']:
            call_prices = self.call_price(S_range, K, r, sigma, T)
            plt.plot(S_range, call_prices, 'b-', label='Call Option')
        
        if option_type in ['put', 'both']:
            put_prices = self.put_price(S_range, K, r, sigma, T)
            plt.plot(S_range, put_prices, 'r-', label='Put Option')
        
        plt.axvline(x=K, color='gray', linestyle='--', label='Strike Price')
        plt.grid(True)
        plt.title(f'Black-Scholes Option Prices (K={K}, r={r}, σ={sigma}, T={T})')
        plt.xlabel('Stock Price')
        plt.ylabel('Option Price')
        plt.legend()
        plt.show()


    def plot_greeks(self,S_range:np.ndarray,K:float,r:float,sigma:float,T:float,option_type:str ='call') -> None:
        """
        Plot greeks for a given range of S and a given strike price K.

        Parameters:
            S_range (np.ndarray): Range of S values
            K (float): Strike price
            r (float): Risk free rate
            sigma (float): Volatility
            T (float): Time to expiration
            option_type (str): Type of option, either 'call' or 'put'
        
        Returns:
            None
        """
        greeks_func = self.call_greeks if option_type == 'call' else self.put_greeks

        #calculate greeks for each stock price
        greeks_data = {
            'delta' : [],
            'gamma' : [],
            'theta' : [],
            'vega' : [],
            'rho' : []
        }

        for S in S_range:
            greeks = greeks_func(S,K,r,sigma,T)
            for key in greeks_data:
                greeks_data[key].append(greeks[key])

        #create subplots
        fig,axs = plt.subplots(3,2,figsize=(15,12))
        axs = axs.flatten()

        #plot each greek
        for i,(greek,values) in enumerate(greeks_data.items()):
            if i<5: #we have 5 greeks
                axs[i].plot(S_range,values)
                axs[i].set_title(f"{greek.capitalize()} - {option_type.capitalize()} Option")
                axs[i].set_xlabel("Stock Price")
                axs[i].set_ylabel(greek.capitalize())
                axs[i].grid(True)
                axs[i].axvline(x=K,color='gray',linestyle='--',label='Strike Price')

        #remove the unused subplots
        fig.delaxes(axs[5])

        plt.tight_layout()
        plt.suptitle(f"Black-Scholes Option prices (K={K},r={r},sigma={sigma},T={T})",y=1.02)
        plt.show()

    def create_option_chain(self, S: float, K_range: np.ndarray, r: float, 
                           sigma: float, T: float) -> pd.DataFrame:
        """
        Create an option chain for a range of strike prices.
        
        Parameters:
        -----------
        S : float
            Stock price
        K_range : np.ndarray
            Range of strike prices
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to maturity in years
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing option chain data
        """
        data = []
        
        for K in K_range:
            call_price = float(self.call_price(S, K, r, sigma, T))
            put_price = float(self.put_price(S, K, r, sigma, T))
            
            call_greeks = self.call_greeks(S, K, r, sigma, T)
            put_greeks = self.put_greeks(S, K, r, sigma, T)
            
            data.append({
                'Strike': K,
                'Call_Price': call_price,
                'Put_Price': put_price,
                'Call_Delta': call_greeks['delta'],
                'Put_Delta': put_greeks['delta'],
                'Call_Gamma': call_greeks['gamma'],
                'Put_Gamma': put_greeks['gamma'],
                'Call_Theta': call_greeks['theta'],
                'Put_Theta': put_greeks['theta'],
                'Call_Vega': call_greeks['vega'],
                'Put_Vega': put_greeks['vega'],
                'Call_Rho': call_greeks['rho'],
                'Put_Rho': put_greeks['rho']
            })
        
        return pd.DataFrame(data)
