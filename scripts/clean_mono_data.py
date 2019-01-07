# -*- coding:utf-8 -*-

from collections import Counter
import fastText as ft
import sys
in_file = sys.argv[1]
out_file = sys.argv[2]
langid_model_file = sys.argv[3]
lang = sys.argv[4]

fr = open(in_file, 'r', encoding='utf-8')
fw = open(out_file, 'w+', encoding='utf-8')

langid_model = ft.load_model(langid_model_file)

idx = 0
passed = 0
for line in fr:
	line_ori = line.strip()
	idx += 1
	
	print('Cleaning', idx, ', the pass ratio is', passed/idx, '.', end = '\r')
	if 'http' in line_ori or '.com' in line_ori:
		continue
	langid_pred = langid_model.predict(line_ori)
	if langid_pred[0][0] != '__label__' + lang or langid_pred[1][0] < 0.75:
		continue
	line_ori_s = line_ori.split(' ')
	len_s = len(line_ori_s)
	if len_s < 3 or len_s > 80:
		continue
	count = Counter(line_ori_s)
	if len_s >= 10 and count.most_common()[0][1] / len_s > 0.25:
		continue

	fw.write(line_ori + '\n')
	passed += 1

print('\nFinished.')
fw.close()
fr.close()
