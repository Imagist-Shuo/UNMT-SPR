#!/bin/bash

# prepare
sudo apt-get install build-essential git-core pkg-config automake libtool wget zlib1g-dev python-dev libbz2-dev
git clone https://github.com/moses-smt/mosesdecoder.git
cd mosesdecoder
make -f contrib/Makefiles/install-dependencies.gmake

# install boost_1_64_0
cd ..
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
tar zxvf boost_1_64_0.tar.gz
cd boost_1_64_0/
./bootstrap.sh
./b2 -j4 --prefix=$PWD --libdir=$PWD/lib64 --layout=system link=static install || echo FAILURE

# install mgiza
cd ..
git clone https://github.com/moses-smt/mgiza.git
cd ~/mgiza/mgizapp
export BOOST_ROOT=~/boost_1_64_0
export BOOST_LIBRARYDIR=$BOOST_ROOT/lib64
cmake .
make
make install
cd ~/mosesdecoder/
mkdir word_align_tools 
cp ~/mgiza/mgizapp/bin/* word_align_tools/ 
cp ~/mgiza/mgizapp/scripts/merge_alignment.py word_align_tools/ 

# install mosesdecoder
./bjam --with-boost=~/boost_1_64_0 --with-mgiza=~/mgiza/mgizapp -j 4
