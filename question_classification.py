#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 13:24
# @Author  : Xavier Byrant
# @FileName: utils.py.py
# @Software: PyCharm
import json
import random
from os.path import join

import torch
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import csv
from model import SimcseModel_T5_CNN
import torch.nn.functional as F
import pandas as pd
from question_classification_train import parse_args

args = parse_args()
pretrain_model_path = args.pretrain_model_path
model_name = pretrain_model_path.split("/")[-1]
pt_path = join(args.output_path, "simcse.pt")
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
device = args.device
model = SimcseModel_T5_CNN(pretrained_model=pretrain_model_path, pooling="cls", dropout=args.dropout).to(device)
model.load_state_dict(torch.load(pt_path))

# 计算文本相似度
def question_cosin_sim(source, target):
	model.eval()
	source = tokenizer(source.strip(), max_length=32, truncation=True, padding='max_length',
					   return_tensors='pt')
	target = tokenizer(target.strip(), max_length=32, truncation=True, padding='max_length',
					   return_tensors='pt')
	with torch.no_grad():
		# source        [batch, 1, seq_len] -> [batch, seq_len]
		source_input_ids = source.get('input_ids').squeeze(1).to(device)
		source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
		if model_name in ["roberta", "T5-base", "Sentence-T5-large", "Sentence-T5-base", "jina"]:
			source_token_type_ids = None
		else:
			source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
		source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
		# target        [batch, 1, seq_len] -> [batch, seq_len]
		target_input_ids = target.get('input_ids').squeeze(1).to(device)
		target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
		if model_name in ["roberta", "T5-base", "Sentence-T5-base", "Sentence-T5-large", "jina"]:
			target_token_type_ids = None
		else:
			target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
		target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
		# concat

		sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
	return abs(sim.item())

def question_hop_num(question, n, k):
	f1 = open('data/webQSP/hop_classification/one_hop.csv', 'r', encoding='utf-8')
	f2 = open('data/webQSP/hop_classification/two_hop.csv', 'r', encoding='utf-8')
	reader1 = csv.reader(f1, delimiter='\t')
	reader2 = csv.reader(f2, delimiter='\t')

	row1 = [row for row in reader1]
	row2 = [row for row in reader2]
	one_hop_rand = random.sample(row1, n)
	two_hop_rand = random.sample(row2, n)
	all = one_hop_rand + two_hop_rand
	for index, item in enumerate(all):
		all[index].extend([question_cosin_sim(question, item[0])])
	all = sorted(all, key=lambda x: x[2], reverse=True)
	# print(all)
	one = 0
	two = 0
	for i in all[:k]:
		if i[1] == '0':
			one += 1
		else:
			two += 1
	f1.close()
	f2.close()
	return [one, two].index(max([one, two]))
if __name__ == '__main__':
	# s1 = "what is the average temperature in phoenix az in december"
	# s2 = ["which author was the daughter of steve perry", "who played denver in four christmases", "who played bilbo in the fellowship of the ring", "who was the secretary of state when andrew jackson was president"]
	# for l in s2:
	# 	print(question_cosin_sim(l, s1))
	print(question_hop_num("where was gabriel faure born", 10, 5))
	# 加载模型

	# model = torch.load('output/supervise/jina-CNN-webQSP-bsz-64-lr-3e-05-drop-0.1-allpath/simcse.pt')
	#
	# # 获取权重的键
	# new_model = {}
	#
	# # 修改键的名称并复制权重
	# for key, value in model.items():
	# 	if "jina" in key:
	# 		new_key = key.replace('jina', 'T5')  # 在这里进行键的修改
	# 		new_model[new_key] = value
	# 	else:
	# 		new_model[key] = value
	#
	# # 保存修改后的模型权重到新的.pt文件
	# torch.save(new_model, 'output/supervise/jina-CNN-webQSP-bsz-64-lr-3e-05-drop-0.1-allpath/simcse_new.pt')
	#
	# model = torch.load('output/supervise/jina-CNN-webQSP-bsz-64-lr-3e-05-drop-0.1-allpath/simcse_new.pt')
	#
	# # 获取权重的键
	# keys = model.keys()
	#
	# # 打印权重的键
	# for key in keys:
	# 	print(key)
