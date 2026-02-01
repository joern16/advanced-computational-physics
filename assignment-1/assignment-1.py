#!/usr/bin/env python3

"""
Numerical evaluation of pi using midpoint-rule quadrature.
...
License: MIT
"""

import time
import argparse
import math


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=1000000, help="Number of points to sample at")
    parser.add_argument("-p", "--parallel", action="store_true", help="Use parallel computation")

    return parser.parse_args()


def integrant(x):
    """Integrant function for pi calculation."""
    return 4.0 * math.sqrt(1.0 - x**2)


def serial_integration(start, stop, N):
    """Perform serial midpoint-rule integration."""
    integral = 0.0
    for i in range(start, stop):
        integral += integrant((i + 0.5) / N) / N

    return integral




if __name__ == "__main__":
    args = parse_arguments()
    N = args.N

    if not args.parallel:
        start_time = time.time()
        integral = serial_integration(0, N, N)
        end_time = time.time()
        print(f"Serial integration result: {integral}")
        print(f"Time taken: {end_time - start_time} seconds")
