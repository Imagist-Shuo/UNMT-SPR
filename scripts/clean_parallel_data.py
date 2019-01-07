# -*- coding:utf-8 -*-

from collections import Counter
import fastText as ft
import sys
src_file = sys.argv[1]
trg_file = sys.argv[2]

langid_model_file = sys.argv[3]
src_lang = sys.argv[4]
trg_lang = sys.argv[5]
is_bpe = bool(sys.argv[6])
max_length = int(sys.argv[7])

sfr = open(src_file, 'r', encoding='utf-8')
tfr = open(trg_file, 'r', encoding='utf-8')

sfw = open(src_file + '.cl', 'w+', encoding='utf-8')
tfw = open(trg_file + '.cl', 'w+', encoding='utf-8')

langid_model = ft.load_model(langid_model_file)

idx = 0
passed = 0
while idx < max_length:
	sline_ori = sfr.readline().strip()
	tline_ori = tfr.readline().strip()
	idx += 1
	if not sline_ori or not tline_ori:
		continue
	
	print('Cleaning', idx, ', the pass ratio is', passed/idx, '.', end = '\r')

	if is_bpe:
		sline = sline_ori.replace('@@ ', '')
		tline = tline_ori.replace('@@ ', '')
	else:
		sline = sline_ori
		tline = tline_ori
	if '<UNK>' in tline:
		continue
	langid_pred = langid_model.predict(tline)
	if langid_pred[0][0] != '__label__' + trg_lang or langid_pred[1][0] < 0.75:
		continue
	split_s = sline.split(' ')
	split_t = tline.split(' ')
	len_s = len(split_s)
	len_t = len(split_t)
	if len_s < 5 or len_s > 80 or len_t < 5 or len_t > 80:
		continue
	count = Counter(split_t)
	if len_t >= 10 and count.most_common()[0][1] / len_t > 0.25:
		continue
	if len(split_s) / len(split_t) <= 0.6 or len(split_t) / len(split_s) <= 0.6:
		continue

	sfw.write(sline_ori + '\n')
	tfw.write(tline_ori + '\n')
	passed += 1

print('\nFinished.')
sfw.close()
tfw.close()

sfr.close()
tfr.close()
