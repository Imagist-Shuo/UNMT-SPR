#!/bin/bash
set -e

DATA_PATH=/home/v-shure/UNMT-SPR # your project directory
RUN_T2T=$DATA_PATH/t2tlight/
MOSES_PATH=~/mosesdecoder

MODE=$1
ROUND=$2
LANG_SRC=$3
LANG_TRG=$4
SHARE_BPE=$5
DEVISES=$6

#echo $MODE $ROUND $LANG_SRC $LANG_TRG $SHARE_BPE $DEVISES

LANG_DIRECTION=$LANG_SRC-$LANG_TRG
if [ $LANG_TRG == "en" ]
then
LANG_PAIR=$LANG_TRG-$LANG_SRC
else
LANG_PAIR=$LANG_SRC-$LANG_TRG
fi

if [ $SHARE_BPE == "true" ]
then
VOCAB_SRC=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.vocab
VOCAB_TRG=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.vocab

EMB_SRC=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.vec
EMB_TRG=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.vec

else
VOCAB_SRC=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.$LANG_SRC.vocab
VOCAB_TRG=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.$LANG_TRG.vocab

EMB_SRC=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.$LANG_SRC.vec
EMB_TRG=$DATA_PATH/Vocab/true.$LANG_PAIR.bpe.$LANG_TRG.vec
fi

MODEL=$DATA_PATH/Models/NMT$ROUND/model_${LANG_DIRECTION}


if [ $MODE == "train" ]
then

TRAIN_SRC=$DATA_PATH/Train/bpe/NMT$ROUND/corpus.$LANG_DIRECTION.bpe.$LANG_SRC
TRAIN_TRG=$DATA_PATH/Train/bpe/NMT$ROUND/corpus.$LANG_DIRECTION.bpe.$LANG_TRG
TRAIN_PARAMS="renew_lr=True,use_pretrained_embedding=False,shared_source_target_embedding=True,device_list=$DEVISES,train_steps=200000,batch_size=4096,save_checkpoint_steps=4000"

echo "==============================================================="
echo "Source file:" $TRAIN_SRC
echo "Target file:" $TRAIN_TRG
echo "Vocab files:" $VOCAB_SRC $VOCAB_TRG
echo "Embedding files:" $EMB_SRC $EMB_TRG
echo "Model folder:" $MODEL
echo "Training params:" $TRAIN_PARAMS
echo "==============================================================="

python3 $RUN_T2T/scripts/shuffle_dataset.py --input $TRAIN_SRC $TRAIN_TRG

echo "Start Training........"
python3 $RUN_T2T/train.py --input $TRAIN_SRC.shuf $TRAIN_TRG.shuf \
    --output $MODEL \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --embeddings $EMB_SRC $EMB_TRG \
    --parameters=$TRAIN_PARAMS

elif [ $MODE == "test" ]
then

SRC_FILE=$7
TRANS_FILE=$SRC_FILE.trans

TEST_PARAMS="use_pretrained_embedding=False,device_list=$DEVISES,decode_batch_size=64,beam_size=8,decode_alpha=0.6"

echo "==============================================================="
echo "Source file:" $SRC_FILE
echo "Target file:" $TRANS_FILE
echo "Vocab files:" $VOCAB_SRC $VOCAB_TRG
echo "Model folder:" $MODEL
echo "Test params:" $TEST_PARAMS
echo "==============================================================="

echo "Start Testing........."
python3 $RUN_T2T/translate.py --input $SRC_FILE \
    --output $TRANS_FILE \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --models $MODEL \
    --parameters=$TEST_PARAMS

elif [ $MODE == "joint" ]
then

BATCHSIZE=$7
STARTBATCH=$8
MAX_ITER=$9
MIX_SMT=${10}
SPLIT_CHOICE=${11}
EXP_NAME=${12}

OPPO_DIRECTION=$LANG_TRG-$LANG_SRC

SRC_FILE=$DATA_PATH/Train/bpe/corpus.$LANG_PAIR.bpe.$LANG_SRC.split$SPLIT_CHOICE
OPPO_SRC=$DATA_PATH/Train/bpe/corpus.$LANG_PAIR.bpe.$LANG_TRG.split$SPLIT_CHOICE

TRAIN_PARAMS="renew_lr=True,use_pretrained_embedding=False,shared_source_target_embedding=True,device_list=$DEVISES,train_steps=8000,batch_size=4096,save_checkpoint_steps=4000"
TEST_PARAMS="use_pretrained_embedding=False,device_list=$DEVISES,decode_batch_size=64,beam_size=8,decode_alpha=0.6"

echo "==============================================================="
echo "Source file:" $SRC_FILE
echo "Oppo file:" $OPPO_SRC
echo "Vocab files:" $VOCAB_SRC $VOCAB_TRG
echo "Model folder:" $MODEL
echo "Train params:" $TRAIN_PARAMS
echo "Test params:" $TEST_PARAMS
echo "==============================================================="


LINECOUNT=$(cat $SRC_FILE | wc -l)
TIMES=$(expr $LINECOUNT / $BATCHSIZE)

if [ $MIX_SMT == "true" ]
then
TRAIN_SRC_SMT=$DATA_PATH/Train/bpe/NMT$ROUND/corpus.$LANG_DIRECTION.bpe.$LANG_SRC
TRAIN_TRG_SMT=$DATA_PATH/Train/bpe/NMT$ROUND/corpus.$LANG_DIRECTION.bpe.$LANG_TRG

SMT_LINECOUNT=$(cat $TRAIN_SRC_SMT | wc -l)
SMT_TIMES=$(expr $SMT_LINECOUNT / $BATCHSIZE)
j=$2
fi

for((i=$8;i<$MAX_ITER;i++));
do

if [ $i -ge $TIMES ]
then
let i=0
fi

TRAIN_SRC=$DATA_PATH/Train/bpe/NMT$ROUND/joint.${LANG_DIRECTION}.bpe.$LANG_SRC.$EXP_NAME.round$i
TRAIN_TRG=$DATA_PATH/Train/bpe/NMT$ROUND/joint.${LANG_DIRECTION}.bpe.$LANG_TRG.$EXP_NAME.round$i

if [ ! -f $TRAIN_SRC.shuf ]
then

TMP_FILE=$SRC_FILE.tmp.$i
TRANS_FILE=$SRC_FILE.tmp.$i.trans
if [ ! -f TRANS_FILE ]
then
START=$(expr $i \* $BATCHSIZE + 1)
END=$(expr $START + $BATCHSIZE - 1)

echo "==============================================================="
echo ${START}, ${END}, $TMP_FILE, $TRANS_FILE
sed -n "${START},${END}p" $SRC_FILE > $TMP_FILE

echo 'Translate' $i '...'
echo "==============================================================="
python3 $RUN_T2T/translate.py --input $TMP_FILE \
    --output $TRANS_FILE \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --models $MODEL \
    --parameters=$TEST_PARAMS
fi

mv $TMP_FILE $TMP_FILE.finished
mv $TRANS_FILE $TRANS_FILE.finished

TRAIN_SRC_TMP=$OPPO_SRC.tmp.$i.trans.finished
TRAIN_TRG_TMP=$OPPO_SRC.tmp.$i.finished

echo "==============================================================="
echo 'Wait opposite translated file...'
echo "==============================================================="
while true
do
if [ -f $TRAIN_SRC_TMP ]
then
break
fi
done

mv $TRAIN_SRC_TMP $TRAIN_SRC
mv $TRAIN_TRG_TMP $TRAIN_TRG

if [ $MIX_SMT == "true" ]
then
if [ $j -ge $SMT_TIMES ]
then
let j=0
fi

SMT_START=$(expr $j \* $BATCHSIZE + 1)
SMT_END=$(expr $SMT_START + $BATCHSIZE - 1)
sed -n "${SMT_START},${SMT_END}p" $TRAIN_SRC_SMT >> $TRAIN_SRC
sed -n "${SMT_START},${SMT_END}p" $TRAIN_TRG_SMT >> $TRAIN_TRG

let j+=1

fi

python3 $RUN_T2T/scripts/shuffle_dataset.py --input $TRAIN_SRC $TRAIN_TRG

fi

echo "==============================================================="
echo 'Training' $i '...'
echo 'Train files:'$TRAIN_SRC.shuf, $TRAIN_TRG.shuf
echo "==============================================================="
python3 $RUN_T2T/train.py --input $TRAIN_SRC.shuf $TRAIN_TRG.shuf \
    --output $MODEL \
    --vocab $VOCAB_SRC $VOCAB_TRG \
    --embeddings $EMB_SRC $EMB_TRG \
    --parameters=$TRAIN_PARAMS

done

fi
