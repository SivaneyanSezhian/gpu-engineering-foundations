import torch
import argparse


def nonnegative_int(value: str) -> int:
    try:
        x_int = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")

    if x_int < 0:
        raise argparse.ArgumentTypeError(f"{x_int} is not a nonnegative integer")
    return x_int


parser = argparse.ArgumentParser()
parser.add_argument("dim", type=nonnegative_int, help="matrices will have size (dim, dim)")
args = parser.parse_args()
n = args.dim

A = torch.rand(size=(n, n))
B = torch.rand(size=(n, n))
C = A @ B

print(C)
