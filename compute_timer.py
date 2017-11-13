"""
Command line utility for comparing CPU and GPU usage.
Discussion Tool for Vector Space Computer
Build-out Class taught by Adam Sponterilli
Nov 16, 2017

Todd Matthews
"""

import argparse
import math
import numpy as np
from numba import vectorize
from timeit import default_timer as timer

# Four test functions compiled with numba vectorize
# Only one is used for a run.

@vectorize(['float32(float32, float32, int32)'], target='cpu')
def run_compute(a, b, math_complexity):
    retval = a
    for i in range(math_complexity):
        retval += math.cos(a+b)
    return retval


@vectorize(['float32(float32, float32, int32)'], target='cuda')
def run_compute_gpu(a, b, math_complexity):
    retval = a
    for i in range(math_complexity):
        retval += math.cos(a+b)
    return retval

@vectorize(['float32(float32, float32)'], target='cpu')
def run_compute_no_for(a, b):
    return a + math.cos(a+b) + math.cos(a+b) + math.cos(a+b) + math.cos(a+b) \
             + math.cos(a + b)+ math.cos(a+b) + math.cos(a+b) + math.cos(a+b) \
             + math.cos(a + b)+ math.cos(a+b)

@vectorize(['float32(float32, float32)'], target='cuda')
def run_compute_gpu_no_for(a, b):
    return a + math.cos(a+b) + math.cos(a+b) + math.cos(a+b) + math.cos(a+b) \
             + math.cos(a + b)+ math.cos(a+b) + math.cos(a+b) + math.cos(a+b) \
             + math.cos(a + b)+ math.cos(a+b)


def main():

    str_desc = "Description: Test GPU vs CPU for increasing complex math."

    parser = argparse.ArgumentParser(description=str_desc)
    parser.add_argument('-g', '--gpu', help='Use the gpu for the computation',
                        action='store_true')
    parser.add_argument('-n', '--no_for_loop', help='Vectorized func without for loop, -m=10',
                        action='store_true')
    parser.add_argument('-s', "--size", type = int,
                        help='Size of array', default=320000000)
    parser.add_argument('-m', "--math_complexity", type = int,
                        help='Math Complexity: number of math ops', default=10)
    args = parser.parse_args()

    math_complexity = args.math_complexity
    if args.no_for_loop:
        math_complexity = 10  # fixed at 10 for vectorized funcs above

    print("Testing compute time for array size = %s, math complexity = %s" %
          (args.size, math_complexity))
    print("  Approx Array Size=%s GBytes"%(args.size*3*4/1E9))

    if args.gpu:
        print("  with GPU")
    else:
        print("  with CPU")
    if args.no_for_loop:
        print("  and without for loops in the vectorized function")

    A = np.ones(args.size, dtype=np.float32)
    B = np.ones(args.size, dtype=np.float32)

    start = timer()

    if args.no_for_loop:
        if args.gpu:
            C = run_compute_gpu_no_for(A, B)
        else:
            C = run_compute_no_for(A, B)
    else:
        if args.gpu:
            C = run_compute_gpu(A, B, math_complexity)
        else:
            C = run_compute(A, B, math_complexity)

    compute_time = timer() - start

    print("Results for Last elements = " + str(C[-5:]))
    print("Compute time = %f seconds" % compute_time)


if __name__ == '__main__':
    main()