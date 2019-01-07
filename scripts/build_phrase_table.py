#-*- coding: utf-8 -*-
from collections import defaultdict
import sys
file1 = sys.argv[1]
file2 = sys.argv[2]
srcf = sys.argv[3]
trgf = sys.argv[4]

st = defaultdict()
ts = defaultdict()

with open(file1, 'r', encoding = 'utf-8') as fr:
	while True:
		line = fr.readline().strip()
		if not line:
			break
		split_line = line.split(' ||||| ')
		source_word = split_line[0].strip()
		if source_word not in st:
			st[source_word] = defaultdict(list)
		target_words = split_line[1].split(' ||| ')
		for t_pair in target_words:
			split_wp = t_pair.split(' @@@ ')
			tw = split_wp[0].strip()
			prob = split_wp[1].strip()
			st[source_word][tw].append(prob)

with open(file2, 'r', encoding = 'utf-8') as fr:
	while True:
		line = fr.readline().strip()
		if not line:
			break
		split_line = line.split(' ||||| ')
		source_word = split_line[0].strip()
		if source_word not in ts:
			ts[source_word] = defaultdict(list)
		target_words = split_line[1].split(' ||| ')
		for t_pair in target_words:
			split_wp = t_pair.split(' @@@ ')
			tw = split_wp[0].strip()
			prob = split_wp[1].strip()
			ts[source_word][tw].append(prob)

for src, trgs in st.items():
	for trg in trgs.keys():
		if trg in ts:
			if src in ts[trg]:
				st[src][trg].append(ts[trg][src][0])


for src, trgs in ts.items():
	for trg in trgs.keys():
		if trg in st:
			if src in st[trg]:
				ts[src][trg].append(st[trg][src][0])


with open(srcf, 'w+', encoding = 'utf-8') as fw:
	for src, trgs in ts.items():
		for trg, prob_list in trgs.items():
			if len(prob_list) == 1:
				prob_list.append('0.0')
			fw.write(trg + ' ||| ' + src + ' ||| ' + prob_list[0] + ' ' + prob_list[0] + ' ' + prob_list[1] + ' ' + prob_list[1] + '\n')

with open(trgf, 'w+', encoding = 'utf-8') as fw:
	for src, trgs in st.items():
		for trg, prob_list in trgs.items():
			if len(prob_list) == 1:
				prob_list.append('0.0')
			fw.write(trg + ' ||| ' + src + ' ||| ' + prob_list[0] + ' ' + prob_list[0] + ' ' + prob_list[1] + ' ' + prob_list[1] + '\n')