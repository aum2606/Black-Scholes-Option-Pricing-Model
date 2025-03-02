# Black-Scholes Option Pricing Model

This project implements the Black-Scholes model for pricing European call and put options in C++. It uses the Eigen library for efficient matrix operations, particularly for vectorized pricing of multiple options simultaneously.

## Features

- Black-Scholes formula for European call and put options
- Calculation of option Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation using Newton-Raphson method
- Vectorized pricing for batch processing using Eigen

## Requirements

- C++17 compatible compiler
- Eigen library (version 3.3 or higher)
- CMake (version 3.10 or higher)

## Building the Project

```bash
mkdir build
cd build
cmake ..
make