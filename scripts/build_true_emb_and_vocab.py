# -*- coding: utf-8 -*-

import numpy as np
import sys

vocab_file = sys.argv[1]
bpe_emb = sys.argv[2]
output_vocab = sys.argv[3]
output_emb = sys.argv[4]

vocab = {}
unk_vec = np.zeros(512)
		
with open(bpe_emb, 'r', encoding='utf-8') as fr:
	line = fr.readline()
	i = 0
	while True:
		line = fr.readline().strip()
		if not line:
			break
		split_line = line.split(' ')
		word = split_line[0]
		vocab[word] = split_line[1:]
		vec = np.array(list(map(float, split_line[1:])))
		unk_vec = i/(i+1) * unk_vec + 1/(i+1)*vec
		i += 1
unk_vec=' '.join(list(map(str, list(unk_vec))))
vocab_file = map(str.strip, open(vocab_file, 'r', encoding='utf-8').readlines())

true_vocab = open(output_vocab, 'w+', encoding='utf-8')
true_embeddings = open(output_emb, 'w+', encoding='utf-8')

for v in vocab_file:
	if v in vocab:
		true_embeddings.write(v + ' ' + ' '.join(vocab[v]) + '\n')
		true_vocab.write(v + '\n')
	else:
		if v == '<BOS>':
			true_vocab.write(v + '\n')
			bos_embs = ' '.join(['0.0' for i in range(512)])
			true_embeddings.write(v + ' ' + bos_embs + '\n')
		elif v == '<EOS>':
			true_vocab.write(v + '\n')
			eos_embs = ' '.join(vocab['</s>'])
			true_embeddings.write(v + ' ' + eos_embs + '\n')
		elif v == '<UNK>':
			true_vocab.write(v + '\n')
			true_embeddings.write(v + ' ' + unk_vec + '\n')

true_vocab.close()
true_embeddings.close()
