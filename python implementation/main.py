import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from black_scholes import BlackScholes

def main():
    """
    Demonstrate the Black-Scholes option pricing model.
    """
    print("Black-Scholes option pricing model")
    print("================================")

    #create a Black-Scholes instance
    bs = BlackScholes()

    #option paramaeters
    S = 100.0 #current stock price
    K = 100.0 #strike price
    r = 0.05 #risk free rate
    sigma = 0.2 #volatility
    T = 1.0 #time to expiration in years

    #calculate option prices and greeks
    call_price = bs.call_price(S, K, r, sigma, T)
    put_price = bs.put_price(S, K, r, sigma, T)
    print("\nOption prices: ")
    print(f"  Call Option Prices: {call_price}")
    print(f"  Put Option Price: {put_price}")

    #calculate and display greeks 
    call_greeks = bs.call_greeks(S,K,r,sigma,T)
    put_greeks = bs.put_greeks(S,K,r,sigma,T)

    print("\nCall option Greeks: ")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value}")
    print("\nPut option Greeks: ")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value}")

    #Demonstrate implied volatility calculation
    print("\nImplied volatility Demonstration: ")
    true_vol = 0.2
    call_price = bs.call_price(S,K,r,true_vol,T)
    put_price = bs.put_price(S,K,r,true_vol,T)

    print(f"  True volatility: {true_vol}")
    print(f"  Call price: {call_price}")
    print(f"  Put price: {put_price}")

    implied_call_vol = bs.implied_volatility_call(call_price,S,K,r,T)
    implied_put_vol = bs.implied_volatility_put(put_price,S,K,r,T)
    print(f"  Implied call volatility: {implied_call_vol}")
    print(f"  Implied put volatility: {implied_put_vol}")

    #Demonstrate vectorized pricing
    print("\nVectorized pricing Demonstration: ")
    stock_prices = np.array([90.0,95.0,100.0,105.0,110.0])
    strike_prices = np.full_like(stock_prices,100.0)

    print(f"  Stock prices: {stock_prices}")

    call_prices = bs.call_price(stock_prices, strike_prices, r, sigma, T)
    put_prices = bs.put_price(stock_prices, strike_prices, r, sigma, T)


    print(f"  Call prices: {call_prices}")
    print(f"  Put prices: {put_prices}")

    #create option chain
    print("\nOption chain Demonstration: ")
    strike_range = np.arange(80,121,5)
    option_chain = bs.create_option_chain(S,strike_range,r,sigma,T)
    print(option_chain)

    #plot option prices
    print("\n Generating options price plot ....")
    stock_range = np.linspace(50,150,100)
    bs.plot_option_prices(stock_range,K,r,sigma,T)

    #plot greeks
    print("\n Generating greeks plot ....")
    bs.plot_greeks(stock_range,K,r,sigma,T,'call')
    bs.plot_greeks(stock_range,K,r,sigma,T,'put')

if __name__ == "__main__":
    main()