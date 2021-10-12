import argparse
from pathlib import Path
from utils import chi2_distance
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", required=True,
        type=Path,
        default=Path(__file__).absolute().parent / "data",
        help="Path to the data directory",
    )

    p = parser.parse_args()
    print(p.p, type(p.p))


if __name__ == "__main__":
    main()
