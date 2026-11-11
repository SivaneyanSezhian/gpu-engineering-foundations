import numpy as np
import torch
import argparse
import time


def nonnegative_int(value: str) -> int:
    try:
        x_int = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")

    if x_int < 0:
        raise argparse.ArgumentTypeError(f"{x_int} is not a nonnegative integer")
    return x_int


def positive_int(value: str) -> int:
    try:
        x_int = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")

    if x_int <= 0:
        raise argparse.ArgumentTypeError(f"{x_int} is not a positive integer")
    return x_int


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dim", default=10, type=nonnegative_int, help="matrices will have size (dim, dim)")
parser.add_argument("-r", "--runs", default=100, type=positive_int, help="number of matmul executions")
args = parser.parse_args()
n = args.dim

A = torch.rand(size=(n, n))
B = torch.rand(size=(n, n))

times = np.zeros(shape=args.runs)
for i in range(args.runs):
    before_time = time.time()
    A @ B
    after_time = time.time()
    comp_ns = (after_time - before_time) * 1e9
    times[i] = comp_ns

median_ns = np.median(times)
print(f"size of square matrices: ({n}, {n})")
print(f"median matmul execution time ({args.runs} runs): {median_ns.round(2)} ns")
