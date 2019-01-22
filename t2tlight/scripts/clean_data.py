""" 
Clean Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from langdetect import detect
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle Dataset")

    parser.add_argument("--input", type=str, nargs=2, required=True, help="Dataset")
    parser.add_argument("--lang", type = str, nargs=2, required= True, help="Languages")
    parser.add_argument("--suffix", type=str, default="cl", help="Suffix of Output File")

    return parser.parse_args()

def main(args): 

    with open(args.input[0], "r", errors='ignore') as frs:
        with open(args.input[1], "r", errors='ignore') as frt:
            data1 = [line for line in frs]
            data2 = [line for line in frt]

    if len(data1) != len(data2):
        print(len(data1), len(data2))
        raise ValueError("length of two files are not equal")

    src_lang = args.lang[0]
    trg_lang = args.lang[1]

    with open(args.input[0] + "." + args.suffix, "w") as fws:
        with open(args.input[1] + "." + args.suffix, "w") as fwt:
            for i in range(len(data1)):
                if i % 10000 == 0:
                    print(i, end = '\r')
                try:
                    line1 = data1[i].replace("@@ ", "")
                    line2 = data2[i].replace("@@ ", "")
                    length1 = len(line1.split(' '))
                    length2 = len(line2.split(' '))
                    if length1 / length2 < 0.7 or length2 / length1 < 0.7:
                        continue
                    if detect(line1) != src_lang or detect(line2) != trg_lang:
                        continue
                    fws.write(data1[i])
                    fwt.write(data2[i])
                except:
                    continue

if __name__ == "__main__":
    main(parse_args())