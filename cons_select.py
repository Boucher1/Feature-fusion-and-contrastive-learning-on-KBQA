#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 13:24
# @Author  : Xavier Byrant
# @FileName: utils.py.py
# @Software: PyCharm
import json
from os.path import join

import torch
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import csv
from model import Matching_network, Matching_network_new
import torch.nn.functional as F
import pandas as pd
from cons_select_train import parse_args

args = parse_args()
pretrain_model_path = args.pretrain_model_path
model_name = pretrain_model_path.split("/")[-1]
pt_path = join(args.output_path, "simcse.pt")
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
device = args.device
model = Matching_network_new(pretrained_model=pretrain_model_path, pooling="cls", dropout=args.dropout).to(device)
model.load_state_dict(torch.load(pt_path))

# 计算文本相似度
def cons_cosin_sim(source, target):
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






if __name__ == '__main__':
	s1 = "government.governmental_jurisdiction.governing_officials"
	s2 = ["government.government_position_held.basic_title", "government.government_position_held.governmental_body", "government.government_position_held.office_position_or_title", "government.government_position_held.district_represented",]
	for l in s2:
		print(cons_cosin_sim(s1, l))
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
