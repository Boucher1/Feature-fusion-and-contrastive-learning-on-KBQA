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
from model import SimcseModel, SimcseModel_T5, SimcseModel_T5_CNN, SimcseModel_T5_GRU_CNN_Attention
import torch.nn.functional as F
import pandas as pd
from train import parse_args_train



# 计算文本相似度
def cosin_sim(source, target, args, model):

	# print(f"正在调用cosin_sim，此时kesai为{args.kesai}")
	pretrain_model_path = args.pretrain_model_path
	model_name = pretrain_model_path.split("/")[-1]
	# pt_path = join(args.output_path, "simcse.pt")
	tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
	device = args.device
	# model = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=pretrain_model_path, pooling="cls", kesai=args.kesai,
	# 										 dropout=args.dropout).to(device)
	# model.load_state_dict(torch.load(pt_path))
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

def return_topk_path(path, n=1, k=1):
	# 传入的path参数形式[[path,score]...]
	path = sorted(path, key=lambda x: x[n], reverse=True)
	topk_path = path[0:k]
	return topk_path





if __name__ == '__main__':
	s1 = "what does jamaican people speak"
	s2 = ["location.country.languages_spoken",
	  "location.country.administrative_divisions", "location.country.calling_code",
	  "location.country.fifa_code", "location.country.official_language",
	  "book.book_subject.works"]
	for l in s2:
		print(cosin_sim(s1, l))
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
