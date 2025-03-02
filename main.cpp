#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "black_scholes.h"

void printOptionDetails(double S, double K, double r, double sigma, double T) {
    std::cout << "Option Parameters: " << std::endl;
    std::cout << "Stock Price: " << S << std::endl;
    std::cout << "Strike Price: " << K << std::endl;
    std::cout << "Interest Rate: " << r << std::endl;
    std::cout << "Volatility: " << sigma << std::endl;
    std::cout << "Time to Maturity: " << T << std::endl;
    std::cout << std::endl;
}

void printGreeks(const finance::Greeks& greeks, const std::string& optionType) {
    std::cout << optionType << " Option Greeks: " << std::endl;
    std::cout << "Delta: " << greeks.delta << std::endl;
    std::cout << "Gamma: " << greeks.gamma << std::endl;
    std::cout << "Vega: " << greeks.vega << std::endl;
    std::cout << "Theta: " << greeks.theta << std::endl;
    std::cout << "Rho: " << greeks.rho << std::endl;
    std::cout << std::endl;
}

void demonstrateImpliedVolatility() {
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double trueVol = 0.2;

    //calculate option prices with known volatility
    double callPrice = finance::blackScholesCall(S, K, r, trueVol, T);
    double putPrice = finance::blackScholesPut(S, K, r, trueVol, T);

    std::cout << "Implied volatility: " << std::endl;
    std::cout << "Call Price: " << callPrice << std::endl;
    std::cout << "Put Price: " << putPrice << std::endl;
    std::cout << "True Volatility: " << trueVol << std::endl;

    double impliedCallVol = finance::impliedVolatilityCall(callPrice, S, K, r, T);
    double impliedPutVol = finance::impliedVolatilityPut(putPrice, S, K, r, T);

    std::cout << "Call Implied Volatility: " << impliedCallVol << std::endl;
    std::cout << "Put Implied Volatility: " << impliedPutVol << std::endl;
    std::cout << std::endl;
}

void demonstrateVectorizedPricing() {
    //create vectors of stock prices and strike prices
    Eigen::VectorXd stockPrices(5);
        stockPrices << 90.0, 95.0, 100.0, 105.0, 110.0;

        Eigen::VectorXd strikePrices(5);
        strikePrices << 100.0, 100.0, 100.0, 100.0, 100.0;

    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0;

    std::cout << "Vectorized option pricing: " << std::endl;
    std::cout << " Stock prices: ";
    for (int i = 0; i < stockPrices.size(); ++i) {
        std::cout << stockPrices(i) << " ";
    }
    std::cout << std::endl;

    //calculate call and put prices for all stock prices
    Eigen::VectorXd callPrices = finance::blackScholesCallVectorized(stockPrices, strikePrices, r, sigma, T);
    Eigen::VectorXd putPrices = finance::blackScholesPutVectorized(stockPrices, strikePrices, r, sigma, T);

    std::cout << "  Call prices: ";
    for (int i = 0; i < callPrices.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << callPrices(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "  Put prices: ";
    for (int i = 0; i < putPrices.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << putPrices(i) << " ";
    }
    std::cout << std::endl << std::endl;

}

int main() {
    std::cout << "Black-Scholes Option Pricing model" << std::endl;
    std::cout << "===============" << std::endl << std::endl;

    //option parameters
    double S = 100.0; //current stock price
    double K = 100.0; //strike price
    double r = 0.05; //interest rate
    double sigma = 0.2; //volatility
    double T = 1.0; //time to maturity

    printOptionDetails(S, K, r, sigma, T);

    //calculate option prices

    double callPrice = finance::blackScholesCall(S, K, r, sigma, T);
    double putPrice = finance::blackScholesPut(S, K, r, sigma, T);

    std::cout << "Option Prices: " << std::endl;
    std::cout << "Call Price: " << callPrice << std::endl;
    std::cout << "Put Price: " << putPrice << std::endl;
    std::cout << std::endl;

    //calculate and display Greeks 
    finance::Greeks callGreeks = finance::calculateCallGreeks(S, K, r, sigma, T);
    finance::Greeks putGreeks = finance::calculatePutGreeks(S, K, r, sigma, T);


    printGreeks(callGreeks, "Call");
    printGreeks(putGreeks, "Put");

    //Demonstrate implied volatility
    demonstrateImpliedVolatility();

    //Demonstrate vectorized pricing
    demonstrateVectorizedPricing();

    return 0;
}



