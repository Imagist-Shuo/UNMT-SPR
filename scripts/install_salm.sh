#!/bin/bash

sudo git clone https://github.com/moses-smt/salm.git
cd salm/Distribution/Linux/
make allO64

cd ~/mosesdecoder/contrib/sigtest-filter/
make SALMDIR=~/salm


