from sklearn.preprocessing import normalize
import codecs
import numpy as np
import argparse
import collections
import sys
try:
    import cupy
except ImportError:
    cupy = None


def supports_cupy():
    return cupy is not None

def get_cupy():
    return cupy

BATCH_SIZE = 1000
banned_words = ['</s>']

def read(file, threshold=0, vocabulary=None):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    #matrix = np.empty((count, dim)) if vocabulary is None else []
    matrix = []
    for i in range(count):
        line = file.readline().strip()
        #print(i, line)
        word, vec = line.split(' ', 1)
        if word in banned_words:
            continue
        if vocabulary is None or word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' '))
    return words, np.array(matrix)

def length_normalize(matrix, xp):
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, xp.newaxis]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('--src_embeddings', help='the source language embeddings')
    parser.add_argument('--trg_embeddings', help='the target language embeddings')
    parser.add_argument('--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--vocab_size', default='200000', help='boundary of oov')
    parser.add_argument('--max_choice', default='100', help='choice k nearest translation words')
    parser.add_argument('--lambda_factor', default='20', help='softmax lambda')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, src_matrix = read(srcfile)
    trg_words, trg_matrix = read(trgfile)

    vocab_size = min(len(src_words), int(args.vocab_size))
    print(vocab_size)

    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        src_matrix = xp.asarray(src_matrix)
        trg_matrix = xp.asarray(src_matrix)
    else:
        xp = np

     # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        src_matrix = length_normalize(src_matrix, xp)
        trg_matrix = length_normalize(trg_matrix, xp)

    src_matrix = src_matrix[:vocab_size, :]
    trg_matrix = trg_matrix[:vocab_size, :]

    # Build word to index map
    src_ind2word = {i: word for i, word in enumerate(src_words[:vocab_size])}
    trg_ind2word = {i: word for i, word in enumerate(trg_words[:vocab_size])}

    # Read dictionary and compute coverage
    
    src2trg = collections.defaultdict(set)
    max_k = int(args.max_choice)
    lamb = float(args.lambda_factor)

    for i in range(0, vocab_size, BATCH_SIZE):
        j = min(i + BATCH_SIZE, vocab_size)
        similarities = src_matrix[i:j].dot(trg_matrix.T)
        nn = xp.argsort(similarities, axis=1)[:,-max_k:].tolist()
        for k in range(j-i):
            if src_ind2word[i+k] not in src2trg:
                src2trg[src_ind2word[i + k]] = {}
            sim = similarities[k][nn[k]]
            sim = xp.exp(lamb *sim)
            norm_sim = sim / sim.sum()

            for idx in range(max_k):
                w_idx = nn[k][idx]               
                src2trg[src_ind2word[i + k]][trg_ind2word[w_idx]] = norm_sim[idx]
        print(len(src2trg), end = '\r')

    f = open(args.dictionary, 'w+', encoding=args.encoding, errors='surrogateescape')
    for src, trg in src2trg.items():
        f.write(src + ' ||||| ')
        candidates_list = []
        for word, prob in trg.items():
            candidates_list.append(word + ' @@@ ' + str(prob))
        f.write(' ||| '.join(candidates_list) +'\n')

if __name__ == '__main__':
    main()