#pragma once
#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H 

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace finance {
    //standard normal cumulative distribution function
    double normalCDF(double x);

    //standard normal probability density function
    double normalPDF(double x);

    // Black-scholes formula for European call option
    double blackScholesCall(double S, double K, double r, double sigma, double T);

    //black-scholes formula for european put option
    double blackScholesPut(double S, double K, double r, double sigma, double T);

    //calculate option Greeks
    struct Greeks {
        double delta;
        double gamma;
        double vega;
        double theta;
        double rho;
    };

    //calculate Greeks for call option
    Greeks calculateCallGreeks(double S, double K, double r, double sigma, double T);

    //calculate Greeks for put option
    Greeks calculatePutGreeks(double S, double K, double r, double sigma, double T);

    //Vectorized version for batch processing using Eigen
    Eigen::VectorXd blackScholesCallVectorized(
        const Eigen::VectorXd& S,
        const Eigen::VectorXd& K,
        double r,
        double sigma,
        double T
    );

    Eigen::VectorXd blackScholesPutVectorized(
        const Eigen::VectorXd& S,
        const Eigen::VectorXd& K,
        double r,
        double sigma,
        double T
    );

    // Implied volatility calculation using Newton-Raphsen method
    double impliedVolatilityCall(double marketPrice, double S, double K, double r, double T, double initialGuess = 0.2);
    double impliedVolatilityPut(double marketPrice, double S, double K, double r, double T, double initialGuess = 0.2);
} //namespace finance

#endif // BLACK_SCHOLES_H