# Unsupervised Neural Machine Translation with SMT as Posterior Regularization
This repository contains the implementation of unsupervised machine translation models presented in “Unsupervised Neural Machine Translation with SMT as Posterior Regularization” (AAAI 2019). 
## Abstract
>Without real bilingual corpus available, unsupervised Neural Machine Translation (NMT) typically requires pseudo parallel data generated with the back-translation method for the model training. However, due to weak supervision, the pseudo data inevitably contain noises and errors that will be accumulated and reinforced in the subsequent training process, leading to bad translation performance. To address this issue, we introduce phrase based Statistic Machine Translation (SMT) models which are robust to noisy data, as posterior regularizations to guide the training of unsupervised NMT models in the iterative back-translation process. Our method starts from SMT models built with pre-trained language models and word-level translation tables inferred from cross-lingual embeddings. Then SMT and NMT models are optimized jointly and boost each other incrementally in a unified EM framework. In this way, (1) the negative effect caused by errors in the iterative back-translation process can be alleviated timely by SMT filtering noises from its phrase tables; meanwhile, (2) NMT can compensate for the deficiency of fluency inherent in SMT. Experiments conducted on en-fr and en-de translation tasks show that our method outperforms the strong baseline and achieves new state-of-the-art unsupervised machine translation performance.
## Main idea
![text](https://github.com/Imagist-Shuo/UNMT/blob/master/overview.jpg?raw=true)
&#160; &#160; &#160; &#160; &#160;The whole procedure of our method mainly consists of two parts shown in the left and right of the above figure. Given a language pair X-Y, for **model initialization**, we build two initial SMT models with language models pre-trained using monolingual data, and word translation tables inferred from cross-lingual embeddings. 
&#160; &#160; &#160; &#160; &#160;Then the initial SMT models will generate pseudo data to warm up two NMT models. Note that the NMT models are trained using not only the pseudo data generated by SMT models, but those generated by reverse NMT models with the **iterative back-translation** method. After that, the NMT-generated pseudo data are fed to SMT models. As **posterior regularization (PR)**, SMT models timely filter out noises and infrequent errors by constructing strong phrase tables with good and frequent translation patterns, and then generate denoised pseudo data to guide the subsequent NMT training.
&#160; &#160; &#160; &#160; &#160;Benefiting from that, NMT then produces better pseudo data for SMT to extract phrases of higher quality, meanwhile compensating for the deficiency in smoothness inherent in SMT via back-translation. Those two steps are unified in an EM framework, where NMT and SMT models are trained jointly and boost each other incrementally until final convergence. 
&#160; &#160; &#160; &#160; &#160;The comparison result is in the following table, which significantly outperforms the state-of-the-art UNMT system (Lample et al. 2018).
![text](https://github.com/Imagist-Shuo/UNMT/blob/master/result.jpg?raw=true)

## Dependencies

  - Python3
  - Numpy
  - Tensorflow (currently tested on version 1.6.0)
  - [Moses](https://github.com/moses-smt/mosesdecoder)
  - [Salm](https://github.com/moses-smt/salm) (to filter noises from phrase tables of SMT models)
  - [Subword-nmt](https://github.com/rsennrich/subword-nmt) (for BPE)
  - [fastText](https://github.com/facebookresearch/fastText) (to generate word embeddings)
  - [vecmap](https://github.com/artetxem/vecmap) (newest version, to generate cross-lingual embeddings)

The test environment is Ubuntu 16.04 LTS with CUDA 9.0. You need to install all the dependencies listed above. The toughest ones may be Moses and Salm, so we also provide two scripts (_install_moses.sh_ and _install_salm.sh_) to help you simplify the installation. 

## Steps
Clone the whole project into your local folder as "`~`\/UNMT-SPR" by default. In following instructions, we will take the translation systems between English and French as example. **By default**, Moses, fastText and vecmap are installed in "`~`/mosesdecoder", "`~`/fasttext" and "`~`/vecmap", which are noted by $MOSES_HOME, $FASTTEXT_HOME and $VECMAP_HOME respectively. 
**Moreover**, our method trains SMT models and NMT models alternately, so we will get the following models in order, i.e., SMT0, NMT0, SMT1, NMT1, SMT2, NMT2, ... The numbers (0,1,2) are noted by $EPOCH. According to our experiments, three epochs (0,1,2) are enough to reach convergence.

#### 1. Get and preprocess data
In this step, we will get all the monolingual data for each language as well as the test data of wmt14. Next, the data will be tokenized, truecased and applied BPE codes. We also train language models for each language using KenLM within Moses in this step. Finally, we will get the word embeddings and cross-lingual embeddings using fastText and vecmap respectively. Note that the we learn and apply joint BPE of the two languages. And the BPE embeddings are considered as the cross-lingual ones for the initialization of NMT embedding layers.
Just run the following command to finish this step.
```sh
$ cd UNMT-SPR
$ ./unmt.sh prepare
```
After this step, the architecture of the directory will be changed. Keep in mind that the following steps are based on that. And the whole monolingual data are divided into several files with the suffix of ".splitxx".

#### 2. Generate data with initial SMT models (SMT0)
In the Step 1., we have built initial word-translation tables using cross-lingual word embeddings and trained the language models of both languages. In this step 
In Step 1., you have got word translation tables of both translation directions (i.e., en2fr and fr2en) in the folder “Vocab/”, and KenLM models of both languages in the folder “Models/”.  Then, **go to “Config/SMT0”, modify _en-fr.smt0.ini_ and _fr-en.smt0.ini_ according to the paths to the word-translation tables and language models**. Then run the following command:
```sh
$ cd UNMT-SPR
$ ./unmt.sh smt0
```
This will generate pseudo data with initial SMT models. After being cleaned, the data will be moved into "Train/bpe/NMT0" for the subsequent training.
#### 3. Train NMT models
This step can be divided into three substeps. 
(1) First, train the NMT models using the pseudo data generated by former SMT models. 
```sh
$ cd UNMT-SPR/Config/NMT$EPOCH
$ ./run.sh train $EPOCH $SRC_LANG $TRG_LANG true $GPU_LIST
$ ./run.sh train $EPOCH $TRG_LANG $SRC_LANG true $GPU_LIST
```
For example, if we want to train NMT0 and there are four GPUs in our machine, we can run the following command:
```sh
$ cd UNMT-SPR/Config/NMT0
$ ./run.sh train 0 en fr true [0,1,2,3]
$ ./run.sh train 0 fr en true [0,1,2,3]
```
(2) Second, perform iterative back-translation between two NMT models. This substep needs two processes to finish. 
```sh
$ cd UNMT-SPR/Config/NMT$EPOCH
$ # The follwing two commands should be run simultaneously by two processes
$ ./run.sh joint $EPOCH $SRC_LANG $TRG_LANG true $GPU_LIST_1 $BATCH_SIZE 0 20 false $SPLIT_PART trial # Process one
$ ./run.sh joint $EPOCH $SRC_LANG $TRG_LANG true $GPU_LIST_1 $BATCH_SIZE 0 20 false $SPLIT_PART trial # Process two
```
where $SPLIT_PART means the split monolingual file you choose. ($SPLIT_PART = 00 means you want to use XXX.split00 as the training data in the back-translation procedure), and $BATCH_SIZE means the number of sentences you want to use for each back-translation iteration. Taking NMT0 for example again, if we want to use the ".split00" files and the batch size is set to 200000, just run
```sh
$ cd UNMT-SPR/Config/NMT0
$ # The follwing two commands should be run simultaneously by two processes
$ ./run.sh joint 0 en fr true [0,1] 200000 0 20 false 00 trial # Process one
$ ./run.sh joint 0 fr en true [2,3] 200000 0 20 false 00 trial # Process two
```
(3) Third, generate pseudo data using the above NMT models. We need to translate two split files with $SPLIT_PART_1 and $SPLIT_PART2 as suffix (containing 8000000 sentences in total by default). Just run
```sh
$ cd UNMT-SPR
$ ./unmt.sh nmt_trans $EPOCH $NEXT_EPOCH $SPLIT_PART_1 $SPLIT_PART2
```
Taking NMT0 for example, we need to run:
```sh
$ cd UNMT-SPR
$ ./unmt.sh nmt_trans 0 1 01 02 # note that the chosen split files are different from training.
```
#### 4. Train SMT models
This step can be divided into two substeps.
(1) First, train SMT models using the pseudo data generated by the former NMT models. We use Experiment Management System (EMS) of Moses to simplify the training. 
```sh
$ cd UNMT-SPR
$ ./unmt.sh smt_train 1 # taking SMT1 for example
```
(2) Second, generate pseudo data using the above SMT models. After the first substep, the new phrase tables and reordering tables will be put into the folder "Models/SMT$EPOCH". **Go to “Config/SMT$EPOCH/en2fr” and “Config/SMT$EPOCH/fr2en”, modify moses.tuned.ini according to the paths to the new phrase tables and reordering tables.** Then, run the following command.
```sh
$ cd UNMT-SPR
$ ./unmt.sh smt_trans 1 01 # taking epoch 1 and split file xx.split01 for example.
```
#### 5. Repeate 3. and 4.
## References
Please cite [1] and if you found the resources in this repository useful.
[1] S. Ren*, Z. Zhang*, S. Liu, M. Zhou, S. Ma **Unsupervised Neural Machine Translation with SMT as Posterior Regularization**
```
@inproceedings{ren2018unsupervised,
  title={Unsupervised Neural Machine Translation with SMT as Posterior Regularization},
  author={Ren, Shuo and Zhang, Zhirui and Liu, Shujie and Zhou, Ming and Ma, Shuai},
  booktitle = {AAAI},
  year={2019}
}
```
License
----
See the LICENSE file for more details.


