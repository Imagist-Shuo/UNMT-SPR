set -e

SRC_LANG=en
TRG_LANG=fr
LANG_PAIR=$SRC_LANG-$TRG_LANG
MOSES_HOME=~/mosesdecoder
FASTTEXT_HOME=~/fasttext
VECMAP_HOME=~/vecmap
THREADS=40
GPU_LIST=[0,1,2,3]

TOKENIZER=$MOSES_HOME/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES_HOME/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES_HOME/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES_HOME/scripts/tokenizer/remove-non-printing-char.perl
TRUECASE_TRAIN=$MOSES_HOME/scripts/recaser/train-truecaser.perl
TRUECASER=$MOSES_HOME/scripts/recaser/truecase.perl

SRC_RAW=corpus.$SRC
TRG_RAW=corpus.$TRG
SRC_CLEANED=corpus.cl.$SRC
TRG_CLEANED=corpus.cl.$TRG
SRC_TOK=corpus.tok.$SRC
TRG_TOK=corpus.tok.$TRG
SRC_TC=corpus.tc.$SRC
TRG_TC=corpus.tc.$TRG
SRC_BPE=corpus.$LANG_PAIR.bpe.$SRC
TRG_BPE=corpus.$LANG_PAIR.bpe.$TRG

SRC_DEV_RAW=newstest2013.$SRC
TRG_DEV_RAW=newstest2013.$TRG
SRC_DEV_TOK=dev.$LANG_PAIR.tok.$SRC
TRG_DEV_TOK=dev.$LANG_PAIR.tok.$TRG
SRC_DEV_TC=dev.$LANG_PAIR.tc.$SRC
TRG_DEV_TC=dev.$LANG_PAIR.tc.$TRG
SRC_DEV_BPE=dev.$LANG_PAIR.bpe.$SRC
TRG_DEV_BPE=dev.$LANG_PAIR.bpe.$TRG
SRC_TEST_RAW=newstest2014-fren-src.en
TRG_TEST_RAW=newstest2014-fren-ref.fr
SRC_TEST_TOK=test.$LANG_PAIR.tok.$SRC
TRG_TEST_TOK=test.$LANG_PAIR.tok.$TRG
SRC_TEST_TC=test.$LANG_PAIR.tc.$SRC
TRG_TEST_TC=test.$LANG_PAIR.tc.$TRG
SRC_TEST_BPE=test.$LANG_PAIR.bpe.$SRC
TRG_TEST_BPE=test.$LANG_PAIR.bpe.$TRG

MODE=$1

if [ $MODE == "prepare" ]
then
# create path
mkdir -p Train/tok/SMT1 Train/tok/SMT2 Train/tok/SMT3 Train/bpe/NMT0 Train/bpe/NMT1 Train/bpe/NMT2 Train/bpe/NMT3 Train/bpe/R2L
mkdir -p Models/NMT0 Models/NMT1 Models/NMT2 Models/SMT0 Models/SMT1 Models/SMT2 Models/R2L
mkdir Vocab
mkdir Test

echo "Download training data..."
cd Train
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz
wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz
#wget -c http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz
#wget -c http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz
#wget -c http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz

wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz
wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.fr.shuffled.v2.gz
#wget -c http://data.statmt.org/wmt17/translation-task/news.2015.fr.shuffled.gz
#wget -c http://data.statmt.org/wmt17/translation-task/news.2016.fr.shuffled.gz
#wget -c http://data.statmt.org/wmt17/translation-task/news.2017.fr.shuffled.gz

# Germen data
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz
# wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz
# wget -c http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz
# wget -c http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz

echo "Combining data..."

for FILENAME in news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
  echo "Concatenating monolingual data..."
  cat $(ls news*$SRC* | grep -v gz) > $SRC_RAW
  cat $(ls news*$TRG* | grep -v gz) > $TGT_RAW
fi
echo "Src monolingual data concatenated in: $SRC_RAW"
echo "Trg monolingual data concatenated in: $TGT_RAW"

echo "Cleaning data..."
wget -c https://s3-us-west-1.amazonaws.com/fasttext-vectors/supervised_models/lid.176.bin
mv lid.176.bin ../Models/langid.bin

python3 ~/UNMT-SPR/scripts/clean_mono_data.py $SRC_RAW $SRC_CLEANED ../Models/langid.bin $SRC_LANG
python3 ~/UNMT-SPR/scripts/clean_mono_data.py $TRG_RAW $TRG_CLEANED ../Models/langid.bin $TRG_LANG

echo "Downloading test data..."
cd ../Test
wget -c http://data.statmt.org/wmt17/translation-task/dev.tgz
tar -xzf dev.tgz

echo "Tokenizing training data..."
cd ../Train
cat $SRC_CLEANED | $NORM_PUNC -l $SRC_LANG | $TOKENIZER -l $SRC_LANG -no-escape -threads $THREADS > $SRC_TOK
cat $TRG_CLEANED | $NORM_PUNC -l $TRG_LANG | $TOKENIZER -l $TRG_LANG -no-escape -threads $THREADS > $SRC_TOK

echo "Tokenizing valid and test data..."
cd ../Test/dev
$INPUT_FROM_SGM < $SRC_DEV.sgm | $NORM_PUNC -l $SRC_LANG | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC_LANG -no-escape -threads $THREADS > ../$SRC_DEV_TOK
$INPUT_FROM_SGM < $TGT_DEV.sgm | $NORM_PUNC -l $TRG_LANG | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TRG_LANG -no-escape -threads $THREADS > ../$TGT_DEV_TOK
$INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l $SRC_LANG | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC_LANG -no-escape -threads $THREADS > ../$SRC_TEST_TOK
$INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l $TRG_LANG | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TRG_LANG -no-escape -threads $THREADS > ../$TRG_TEST_TOK

echo "Truecasing data..."
cd ../../Train
$TRUECASE_TRAIN -model ../Models/truecase-model.$SRC_LANG -corpus $SRC_TOK
$TRUECASE_TRAIN -model ../Models/truecase-model.$TRG_LANG -corpus $TRG_TOK
$TRUECASER < $SRC_TOK > $SRC_TC -model ../Models/truecase-model.$SRC_LANG
$TRUECASER < $TRG_TOK > $TRG_TC -model ../Models/truecase-model.$TRG_LANG
cd ../Test
$TRUECASER < $SRC_DEV_TOK > $SRC_DEV_TC -model ../Models/truecase-model.$SRC_LANG
$TRUECASER < $TRG_DEV_TOK > $TRG_DEV_TC -model ../Models/truecase-model.$TRG_LANG
$TRUECASER < $SRC_TEST_TOK > $SRC_TEST_TC -model ../Models/truecase-model.$SRC_LANG
$TRUECASER < $TRG_TEST_TOK > $TRG_TEST_TC -model ../Models/truecase-model.$TRG_LANG

echo "Training BPE codes and applying..."
cd ../Train
subword-nmt learn-joint-bpe-and-vocab --input $SRC_TOK $TRG_TOK -s 60000 -o ../Vocab/$LANG_PAIR.code --write-vocabulary ../Vocab/$LANG_PAIR.bpe.$SRC_LANG.vocab ../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab
subword-nmt apply-bpe -c ../Vocab/$LANG_PAIR.code --vocabulary ../Vocab/$LANG_PAIR.bpe.$SRC_LANG.vocab --vocabulary-threshold 50 < $SRC_TC > $SRC_BPE
subword-nmt apply-bpe -c ../Vocab/$LANG_PAIR.code --vocabulary ../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < $TRG_TC > $TRG_BPE
cd ../Test
subword-nmt apply-bpe -c ../Vocab/$LANG_PAIR.code --vocabulary ../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < $SRC_DEV_TC > $SRC_DEV_BPE
subword-nmt apply-bpe -c ../Vocab/$LANG_PAIR.code --vocabulary ../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < $TRG_DEV_TC > $TRG_DEV_BPE
subword-nmt apply-bpe -c ../Vocab/$LANG_PAIR.code --vocabulary ../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < $SRC_TEST_TC > $SRC_TEST_BPE
subword-nmt apply-bpe -c ../Vocab/$LANG_PAIR.code --vocabulary ../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < $TRG_TEST_TC > $TRG_TEST_BPE

echo "Trainig word embeddings..."
cd ../Train
$FASTTEXT_HOME/fasttext skipgram -input $SRC_TC -output ../Vocab/$SRC_LANG.emb -neg 10 -dim 512 -ws 5 -thread $THREADS
$FASTTEXT_HOME/fasttext skipgram -input $TRG_TC -output ../Vocab/$TRG_LANG.emb -neg 10 -dim 512 -ws 5 -thread $THREADS
echo "Trainig shared bpe embeddings..."
cat $SRC_BPE $TRG_BPE > combined.bpe
$FASTTEXT_HOME/fasttext skipgram -input combined.bpe -output ../Vocab/$LANG_PAIR.bpe.emb -neg 10 -dim 512 -ws 5 -thread $THREADS
rm combined.bpe

echo "Training cross-lingual word embeddings..."
cd ../Vocab
python3 $VECMAP_HOME/map_embeddings.py --unsupervised $SRC_LANG.emb.vec $SRC_LANG.emb.vec $LANG_PAIR.$SRC_LANG.vec $LANG_PAIR.$TRG_LANG.vec

echo "Inferring dictionary..."
python3 ~/UNMT-SPR/scripts/build_dic_from_emb_multichoice.py --src_embeddings $LANG_PAIR.$SRC_LANG.vec --trg_embeddings $LANG_PAIR.$TRG_LANG.vec --dictionary $SRC_LANG-$TRG_LANG.dic \
	--vocab_size 200000 --max_choice 100 --lambda_factor 20 --cuda
python3 ~/UNMT-SPR/scripts/build_dic_from_emb_multichoice.py --src_embeddings $LANG_PAIR.$TRG_LANG.vec --trg_embeddings $LANG_PAIR.$SRC_LANG.vec --dictionary $TRG_LANG-$SRC_LANG.dic \
	--vocab_size 200000 --max_choice 100 --lambda_factor 20 --cuda
python3 ~/UNMT-SPR/scripts/build_phrase_table.py $SRC_LANG-$TRG_LANG.dic $TRG_LANG-$SRC_LANG.dic $SRC_LANG $TRG_LANG

echo "Generate pre-trained embeddings and vocab files for NMT models..."
cat ../Train/$SRC_BPE ../Train/$TRG_BPE > train_data_bpe.whole
echo "Build Vocab........"
python3 ~/UNMT-SPR/scripts/build_vocab.py train_data_bpe.whole train_data_bpe.vocab --vocabsize 60000 
python3 ~/UNMT-SPR/scripts/build_true_emb_and_vocab.py train_data_bpe.vocab $LANG_PAIR.bpe.emb.vec true.$LANG_PAIR.bpe.vocab true.$LANG_PAIR.bpe.vec
rm train_data_bpe.whole train_data_bpe.vocab

echo "Training KenLM..."
cd ../Train
$MOSES_HOME/bin/lmplz -o 5 --prune 0 1 1 2 2 < $SRC_TC > ../Models/lm.arpa.$SRC
$MOSES_HOME/bin/build_binary ../Models/lm.arpa.$SRC ../Models/lm.bin.$SRC
$MOSES_HOME/bin/lmplz -o 5 --prune 0 1 1 2 2 < $TRG_TC > ../Models/lm.arpa.$TRG 
$MOSES_HOME/bin/build_binary ../Models/lm.arpa.$TRG ../Models/lm.bin.$TRG

echo "Spliting files..." # this step will split the whole training data into several files, each of 4,000,000 lines.
mv $SRC_TC ./tok
mv $TRG_TC ./tok
mv $SRC_BPE ./bpe
mv $TRG_BPE ./bpe
cd ./tok
shuf $SRC_TC > $SRC_TC.shuf
rm $SRC_TC
mv $SRC_TC.shuf $SRC_TC
split -a 2 -d -l 4000000 $SRC_TC $SRC_TC.split
shuf $TRG_TC > $TRG_TC.shuf
rm $TRG_TC
mv $TRG_TC.shuf $TRG_TC
split -a 2 -d -l 4000000 $TRG_TC $TRG_TC.split
cd ../bpe
shuf $SRC_BPE > $SRC_BPE.shuf
rm $SRC_BPE
mv $SRC_BPE.shuf $SRC_BPE
split -a 2 -d -l 4000000 $SRC_BPE $SRC_BPE.split
shuf $TRG_BPE > $TRG_BPE.shuf
rm $TRG_BPE
mv $TRG_BPE.shuf $TRG_BPE
split -a 2 -d -l 4000000 $TRG_BPE $TRG_BPE.split

elif [ $MODE == "smt0" ]
then
cd Config/SMT0
$MOSES_HOME/bin/moses -f $SRC_LANG-$TRG_LANG.smt0.ini -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 \
	-threads 40 < ../../Train/$SRC_TC.split00 > ../../Train/$SRC_TC.split00.trans.$TRG_LANG
$MOSES_HOME/bin/moses -f $TRG_LANG-$SRC_LANG.smt0.ini -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 \
	-threads 40 < ../../Train/$TRG_TC.split00 > ../../Train/$TRG_TC.split00.trans.$SRC_LANG
cd ../../Train/tok
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $SRC_TC.split00 $SRC_TC.split00.trans.$TRG_LANG ../../Models/langid.bin $SRC_LANG $TRG_LANG False 4000000
mv $SRC_TC.split00.cl ../bpe/NMT0/corpus.$SRC_LANG-$TRG_LANG.tc.$SRC_LANG
mv $SRC_TC.split00.trans.$TRG_LANG.cl ../bpe/NMT0/corpus.$SRC_LANG-$TRG_LANG.tc.$TRG_LANG
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $TRG_TC.split00 $TRG_TC.split00.trans.$SRC_LANG ../../Models/langid.bin $SRC_LANG $TRG_LANG False 4000000
mv $TRG_TC.split00.cl ../bpe/NMT0/corpus.$TRG_LANG-$SRC_LANG.tc.$TRG_LANG
mv $TRG_TC.split00.trans.$SRC_LANG.cl ../bpe/NMT0/corpus.$TRG_LANG-$SRC_LANG.tc.$SRC_LANG

cd ../bpe/NMT0
subword-nmt apply-bpe -c ../../Vocab/$LANG_PAIR.code --vocabulary ../../Vocab/$LANG_PAIR.bpe.$SRC_LANG.vocab --vocabulary-threshold 50 < corpus.$SRC_LANG-$TRG_LANG.tc.$SRC_LANG > corpus.$SRC_LANG-$TRG_LANG.bpe.$SRC_LANG
subword-nmt apply-bpe -c ../../Vocab/$LANG_PAIR.code --vocabulary ../../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < corpus.$SRC_LANG-$TRG_LANG.tc.$TRG_LANG > corpus.$SRC_LANG-$TRG_LANG.bpe.$TRG_LANG

elif [ $MODE == "nmt_trans" ]
# Translating
EPOCH=$2
NEXT_EPOCH=$3
SPLIT_PART_1=$4
SPLIT_PART_2=$5
./run.sh test $EPOCH $SRC_LANG $TRG_LANG true $GPU_LIST ../../Train/bpe/$SRC_BPE.split$SPLIT_PART_1
./run.sh test $EPOCH $SRC_LANG $TRG_LANG true $GPU_LIST ../../Train/bpe/$SRC_BPE.split$SPLIT_PART_2
./run.sh test $EPOCH $TRG_LANG $SRC_LANG true $GPU_LIST ../../Train/bpe/$TRG_BPE.split$SPLIT_PART_1
./run.sh test $EPOCH $TRG_LANG $SRC_LANG true $GPU_LIST ../../Train/bpe/$TRG_BPE.split$SPLIT_PART_2

cd ../../Train/bpe
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $SRC_BPE.split$SPLIT_PART_1 $SRC_BPE.split$SPLIT_PART_1.trans ../../Models/langid.bin $SRC_LANG $TRG_LANG True 4000000
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $SRC_BPE.split$SPLIT_PART_2 $SRC_BPE.split$SPLIT_PART_2.trans ../../Models/langid.bin $SRC_LANG $TRG_LANG True 4000000
cat $SRC_BPE.split$SPLIT_PART_1.cl $SRC_BPE.split$SPLIT_PART_2.cl > ../tok/SMT$NEXT_EPOCH/corpus.$TRG_LANG-$SRC_LANG.bpe.$SRC_LANG
cat $SRC_BPE.split$SPLIT_PART_1.trans.cl $SRC_BPE.split$SPLIT_PART_2.trans.cl > ../tok/SMT$NEXT_EPOCH/corpus.$TRG_LANG-$SRC_LANG.bpe.$TRG_LANG
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $TRG_BPE.split$SPLIT_PART_1 $TRG_BPE.split$SPLIT_PART_1.trans ../../Models/langid.bin $TRG_LANG $SRC_LANG True 4000000
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $TRG_BPE.split$SPLIT_PART_2 $TRG_BPE.split$SPLIT_PART_2.trans ../../Models/langid.bin $TRG_LANG $SRC_LANG True 4000000
cat $TRG_BPE.split$SPLIT_PART_1.cl $TRG_BPE.split$SPLIT_PART_2.cl > ../tok/SMT$NEXT_EPOCH/corpus.$SRC_LANG-$TRG_LANG.bpe.$TRG_LANG
cat $TRG_BPE.split$SPLIT_PART_1.trans.cl $TRG_BPE.split$SPLIT_PART_2.trans.cl > ../tok/SMT$NEXT_EPOCH/corpus.$SRC_LANG-$TRG_LANG.bpe.$SRC_LANG
cd ../tok/SMT$NEXT_EPOCH
sed 's/@@ //g' corpus.$SRC_LANG-$TRG_LANG.bpe.$SRC_LANG > corpus.$SRC_LANG-$TRG_LANG.tc.$SRC_LANG
sed 's/@@ //g' corpus.$SRC_LANG-$TRG_LANG.bpe.$TRG_LANG > corpus.$SRC_LANG-$TRG_LANG.tc.$TRG_LANG
sed 's/@@ //g' corpus.$TRG_LANG-$SRC_LANG.bpe.$TRG_LANG > corpus.$TRG_LANG-$SRC_LANG.tc.$TRG_LANG
sed 's/@@ //g' corpus.$TRG_LANG-$SRC_LANG.bpe.$SRC_LANG > corpus.$TRG_LANG-$SRC_LANG.tc.$SRC_LANG

elif [ $MODE == "smt_train" ]
then
EPOCH=$2
mkdir -p UNMT-SPR/SMT_WORKPLACE/SMT_$EPOCH/$SRC_LANG-$TRG_LANG UNMT-SPR/SMT_WORKPLACE/SMT_$EPOCH/$TRG_LANG-$SRC_LANG
cd Config/SMT$EPOCH
# Training
cd $SRC_LANG2$TRG_LANG
~/mosesdecoder/scripts/ems/experiment.perl -config $SRC_LANG2$TRG_LANG.salm.config -exec &> ../$SRC_LANG2$TRG_LANG.log
cd $TRG_LANG2$SRC_LANG
~/mosesdecoder/scripts/ems/experiment.perl -config $TRG_LANG2$SRC_LANG.salm.config -exec &> ../$TRG_LANG2$SRC_LANG.log
cd ../../
cp SMT_WORKPLACE/SMT_$EPOCH/$SRC_LANG-$TRG_LANG/tuning/moses.tuned.ini.1 Config/SMT$EPOCH/$SRC_LANG2$TRG_LANG/moses.tuned.ini
cp SMT_WORKPLACE/SMT_$EPOCH/$SRC_LANG-$TRG_LANG/model/phrase-table-sigtest-filter.1.gz Models/SMT$EPOCH/phrase-table.$SRC_LANG2$TRG_LANG.gz
cp SMT_WORKPLACE/SMT_$EPOCH/$SRC_LANG-$TRG_LANG/model/reordering-table-sigtest-filter.1.wbe-msd-bidirectional-fe.gz Models/SMT$EPOCH/reordering-table.$SRC_LANG2$TRG_LANG.gz
cp SMT_WORKPLACE/SMT_$EPOCH/$TRG_LANG-$SRC_LANG/tuning/moses.tuned.ini.1 Config/SMT$EPOCH/$TRG_LANG2$SRC_LANG/moses.tuned.ini
cp SMT_WORKPLACE/SMT_$EPOCH/$TRG_LANG-$SRC_LANG/model/phrase-table-sigtest-filter.1.gz Models/SMT$EPOCH/phrase-table.$TRG_LANG2$SRC_LANG.gz
cp SMT_WORKPLACE/SMT_$EPOCH/$TRG_LANG-$SRC_LANG/model/reordering-table-sigtest-filter.1.wbe-msd-bidirectional-fe.gz Models/SMT$EPOCH/reordering-table.$TRG_LANG2$SRC_LANG.gz

elif [ $MODE == "smt_trans" ]
then
EPOCH=$2
SPLIT_PART=$3
cd Config/SMT$EPOCH/$SRC2$TRG
$MOSES_HOME/bin/moses -f moses.tuned.ini -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 \
	-threads 40 < ../../../Train/$SRC_TC.split$SPLIT_PART > ../../../Train/$SRC_TC.split$SPLIT_PART.trans.$TRG_LANG
cd ../$TRG2$SRC
$MOSES_HOME/bin/moses -f $TRG_LANG-$SRC_LANG.smt0.ini -search-algorithm 1 -cube-pruning-pop-limit 5000 -s 5000 \
	-threads 40 < ../../../Train/$TRG_TC.split$SPLIT_PART > ../../../Train/$TRG_TC.split$SPLIT_PART.trans.$SRC_LANG
cd ../../Train/tok
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $SRC_TC.split$SPLIT_PART $SRC_TC.split$SPLIT_PART.trans.$TRG_LANG ../../Models/langid.bin $SRC_LANG $TRG_LANG False 4000000
mv $SRC_TC.split$SPLIT_PART.cl ../bpe/NMT$EPOCH/corpus.$SRC_LANG-$TRG_LANG.tc.$SRC_LANG
mv $SRC_TC.split$SPLIT_PART.trans.$TRG_LANG.cl ../bpe/NMT$EPOCH/corpus.$SRC_LANG-$TRG_LANG.tc.$TRG_LANG
python3 ~/UNMT-SPR/scripts/clean_parallel_data.py $TRG_TC.split$SPLIT_PART $TRG_TC.split$SPLIT_PART.trans.$SRC_LANG ../../Models/langid.bin $SRC_LANG $TRG_LANG False 4000000
mv $TRG_TC.split$SPLIT_PART.cl ../bpe/NMT$EPOCH/corpus.$TRG_LANG-$SRC_LANG.tc.$TRG_LANG
mv $TRG_TC.split$SPLIT_PART.trans.$SRC_LANG.cl ../bpe/NMT$EPOCH/corpus.$TRG_LANG-$SRC_LANG.tc.$SRC_LANG

cd ../bpe/NMT$EPOCH
subword-nmt apply-bpe -c ../../Vocab/$LANG_PAIR.code --vocabulary ../../Vocab/$LANG_PAIR.bpe.$SRC_LANG.vocab --vocabulary-threshold 50 < corpus.$SRC_LANG-$TRG_LANG.tc.$SRC_LANG > corpus.$SRC_LANG-$TRG_LANG.bpe.$SRC_LANG
subword-nmt apply-bpe -c ../../Vocab/$LANG_PAIR.code --vocabulary ../../Vocab/$LANG_PAIR.bpe.$TRG_LANG.vocab --vocabulary-threshold 50 < corpus.$SRC_LANG-$TRG_LANG.tc.$TRG_LANG > corpus.$SRC_LANG-$TRG_LANG.bpe.$TRG_LANG

fi

