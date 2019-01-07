""" 
Clean Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from langdetect import detect
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Dataset")

    parser.add_argument("--input", type=str, required=True, help="Dataset")
    parser.add_argument("--output", type=str, required=True, help="Dataset")

    return parser.parse_args()

def main(args): 

    with open(args.input, "r", encoding = 'UTF8', errors='ignore') as fr:
        with open(args.output, "w+", encoding = 'UTF8', errors='ignore') as fw:
            while True:
                line = fr.readline().strip()
                if not line:
                    break
                split_line = line.split(' ')
                if len(split_line) < 5 or len(split_line) > 80:
                    continue
                fw.write(line + '\n')


if __name__ == "__main__":
    main(parse_args())