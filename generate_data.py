#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/21 19:20
# @Author  : Xavier Byrant
# @FileName: generate_data.py
# @Software: PyCharm
import json
import torch.nn.functional as F
import os
import random

# 下一步工作，把新生成的训练数据加上主题实体拿去训练
import pandas as pd
import csv
from os.path import join
from train import parse_args_train
from tqdm import tqdm

from query_virtuoso import query_virtuoso_entityType, get_1hop_p, get_2hop_p, get_candid_entitycons, get_candid_con_path
from transformers import BertTokenizer, AutoTokenizer
from utils import cosin_sim
# 合并字符串
from collections import OrderedDict

args = parse_args_train()
tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)

"""没有引入查询不完全的查询图时的数据处理"""
# 生成训练集的格式(sent0,sent1,hard_neg)
def generate_train_data(in_file, out_file):
	df = pd.read_csv(in_file, sep='\t')
	new_question = ''
	pos = []
	new_rows = []
	for index, row in df.iterrows():
		quesion = row['sent0']
		query_path = row['sent1']
		label = row['label']
		if quesion != new_question or label == 1:
			if quesion != new_question:
				new_question = quesion
			if row['label'] == 1:
				pos = row['sent1']
			continue
		new_line = [quesion, pos, query_path]
		new_rows.append(new_line)
	with open(out_file, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(['sent0', 'sent1', 'hard_neg'])
		writer.writerows(new_rows)


# 根据问句和查询路径相似度得分来生成
def create_topk_data(path):
	train = open(f"data/webQSP/mask_data/SBERT_{path}_topK_NoCons_qtype.csv", "w", encoding="utf-8", newline="")
	train_w = csv.writer(train, delimiter='\t')
	# train = open(f"data/SBERT_test_topK_NoCons.csv", "w", encoding="utf-8")
	# train_w = csv.writer(train, delimiter='\t')
	"""只对主路径进行打分选择"""

	num = 1
	K = 5  # top_k 负样本

	with open(f"data/webQSP/source_data/WebQSP.{path}.json", "r", encoding="utf-8") as f:
		# with open(f"../../Datasets/WQSP/test.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in tqdm(data["Questions"]):
			"""获取基本信息"""
			query = i["ProcessedQuestion"]
			mention = i["Parses"][0]["PotentialTopicEntityMention"]
			if mention is None:
				mention = ""
			# query_e = query.replace(mention, "<e>")
			query_e = query
			TopicEntityMid = i["Parses"][0]["TopicEntityMid"]
			# 没有answer就跳过
			try:
				AnswerEntityMid = i["Parses"][0]["Answers"][0]["AnswerArgument"]
			except Exception as e:
				print(e)
				continue
			InferentialChain = i["Parses"][0]["InferentialChain"]

			"""生成训练数据"""
			if InferentialChain is not None:
				# 主题实体类型
				top_e_typename = query_virtuoso_entityType(TopicEntityMid)
				# 答案实体类型
				answer_typename = query_virtuoso_entityType(AnswerEntityMid)
				# 要是查不到类型就跳过
				if top_e_typename is None or answer_typename is None:
					continue
				query_e = query_e + ' [unused0] ' + answer_typename
				# 此处将正样本加上问句主题实体的类型名字
				core_path_pos = InferentialChain[0] + ' [unused0] ' + top_e_typename
				train_w.writerow([query_e, core_path_pos, 1])  # 写入主路径的正样本
				# print(query_e, core_path_pos, 1)
				num += 1
				pos_g = tokenizer(core_path_pos, padding='max_length', max_length=100, truncation=True,
								  return_tensors="pt")['input_ids'].float()
				Scores_neg = {}  # 负样本打分
				one_hop_path = get_1hop_p(TopicEntityMid)

				if one_hop_path:
					for p in one_hop_path:
						if p not in InferentialChain:  # path不是正样本
							# 此处将负样本加上问句主题实体的类型名字
							core_path_neg = p + ' [unused0] ' + top_e_typename
							neg_g = tokenizer(core_path_neg, padding='max_length', max_length=100, truncation=True,
											  return_tensors="pt")['input_ids']
							Score = F.cosine_similarity(pos_g, neg_g, dim=1)[0].item()
							Scores_neg[p] = Score
				# 倒序  选择Top_K负样本
				reverse_Score_neg = sorted(Scores_neg.items(), key=lambda item: item[1], reverse=True)
				num_one_hop = 1

				for i in reverse_Score_neg:
					if i[0] != InferentialChain[0]:  # 剔除正样本
						core_path_neg = i[0] + ' [unused0] ' + top_e_typename
						num_one_hop += 1
						train_w.writerow([query_e, core_path_neg, 0])
						# print(query_e, core_path_neg, 0)
						num += 1
						if num_one_hop > K:
							break
				# continue

				if len(InferentialChain) == 2:
					# print("-" * 50)
					Scores_neg2 = {}
					# 若存在两跳的路径 则更新正样本 否则使用第一跳的正样本进行选择最相似的作为负样本
					# p_info.append(InferentialChain[1])
					core_path_pos = InferentialChain[1] + ' [unused0] ' + top_e_typename
					train_w.writerow([query_e, core_path_pos, 1])  # 写入主路径的正样本
					# print(query_e, core_path_pos, 1)
					num += 1
					pos_g = tokenizer(core_path_pos, padding='max_length', max_length=100, truncation=True,
									  return_tensors="pt")['input_ids'].float()
					# 负样本来源于第一跳正确但是第二跳不正确的样本
					two_hop_path = get_2hop_p(TopicEntityMid, InferentialChain[0])
					if two_hop_path:
						for p2 in two_hop_path:
							if p2 not in InferentialChain:  # path不是正样本
								core_path_neg = i[0] + ' [unused0] ' + top_e_typename
								neg_g2 = tokenizer(core_path_neg, padding='max_length', max_length=100, truncation=True,
												   return_tensors="pt")['input_ids']
								Score2 = F.cosine_similarity(pos_g, neg_g2, dim=1)[0].item()
								Scores_neg2[p2] = Score2
					# 倒序  选择Top_K负样本
					reverse_Score_neg_2hop = sorted(Scores_neg2.items(), key=lambda item: item[1], reverse=True)
					num_two_hop = 1
					for i in reverse_Score_neg_2hop:
						if i[0] not in InferentialChain:  # 剔除正样本
							# 加入修改后的负样本
							core_path_neg = i[0] + ' [unused0] ' + top_e_typename
							num_two_hop += 1
							train_w.writerow([query_e, core_path_neg, 0])
							# print(query_e, i[0], 0)
							num += 1
							if num_two_hop > K:
								break
					continue


# 根据topic id生成所有路径
def create_data_allpath(path):
	train = open(f"data/webQSP/mask_data/SBERT_{path}_allpath_NoCons.csv", "w", encoding="utf-8", newline="")
	train_w = csv.writer(train, delimiter='\t')
	# train = open(f"data/SBERT_test_topK_NoCons.csv", "w", encoding="utf-8")
	# train_w = csv.writer(train, delimiter='\t')
	"""只对主路径进行打分选择"""

	num = 0
	K = 5  # top_k 负样本

	with open(f"data/webQSP/source_data/WebQSP.{path}.json", "r", encoding="utf-8") as f:
		# with open(f"../../Datasets/WQSP/test.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in data["Questions"]:
			"""获取基本信息"""
			query = i["ProcessedQuestion"]
			mention = i["Parses"][0]["PotentialTopicEntityMention"]
			if mention is None:
				mention = ""
			query_e = query.replace(mention, "<e>")
			# query_e = query
			TopicEntityMid = i["Parses"][0]["TopicEntityMid"]
			InferentialChain = i["Parses"][0]["InferentialChain"]
			# 没有answer就跳过
			try:
				AnswerEntityMid = i["Parses"][0]["Answers"][0]["AnswerArgument"]
			except Exception as e:
				print(e)
				continue
			"""生成训练数据"""
			# p_info = []
			if InferentialChain is not None:
				num += 1
				if path == "test":
					train_w.writerow([query_e, InferentialChain[0], 1])  # 写入主路径的正样本
					print(query_e, InferentialChain[0], 1)
					one_hop_path = get_1hop_p(TopicEntityMid)
					if one_hop_path:
						for p in one_hop_path:
							if p not in InferentialChain:  # 剔除正样本
								train_w.writerow([query_e, p, 0])
								print(query_e, p, 0)
					if len(InferentialChain) == 2:
						train_w.writerow([query_e, InferentialChain[1], 1])  # 写入主路径的正样本
						print(query_e, InferentialChain[1], 1)
						two_hop_path = get_2hop_p(TopicEntityMid, InferentialChain[0])
						if two_hop_path:
							for p2 in two_hop_path:
								if p2 not in InferentialChain:  # 剔除正样本
									train_w.writerow([query_e, p2, 0])
									print(query_e, p2, 0)
				elif path == "train":
					# 主题实体类型
					# top_e_typename = query_virtuoso_entityType(TopicEntityMid)
					# # 答案实体类型
					# answer_typename = query_virtuoso_entityType(AnswerEntityMid)
					# # 要是查不到类型就跳过
					# if top_e_typename is None or answer_typename is None:
					# 	continue
					# query_e = query_e + ' [unused0] ' + answer_typename
					# core_path_pos = InferentialChain[0] + ' [unused0] ' + top_e_typename
					# p_info.append(InferentialChain[0])
					train_w.writerow([query_e, InferentialChain[0], 1])  # 写入主路径的正样本
					print(query_e, InferentialChain[0], 1)
					one_hop_path = get_1hop_p(TopicEntityMid)
					if one_hop_path:
						for p in one_hop_path:
							if p not in InferentialChain:  # 剔除正样本
								# core_path_neg = p + ' [unused0] ' + top_e_typename
								train_w.writerow([query_e, p, 0])
								print(query_e, p, 0)
					if len(InferentialChain) == 2:
						# core_path_pos = InferentialChain[1] + ' [unused0] ' + top_e_typename
						train_w.writerow([query_e, InferentialChain[1], 1])  # 写入主路径的正样本
						print(query_e, InferentialChain[1], 1)
						two_hop_path = get_2hop_p(TopicEntityMid, InferentialChain[0])
						if two_hop_path:
							for p2 in two_hop_path:
								if p2 not in InferentialChain:  # 剔除正样本
									# core_path_neg = p2 + ' [unused0] ' + top_e_typename
									train_w.writerow([query_e, p2, 0])
									print(query_e, p2, 0)
		print(f"数量有{num}")

# 根据topic id生成所有路径,两跳就两个路径,除了inferetial chain其他的都是负样本
def create_data_beam_search_top_k(path, k):
	train = open(f"data/webQSP/beam_search/{path}_top_k_NoCons.csv", "w", encoding="utf-8", newline="")
	train_w = csv.writer(train, delimiter='\t')
	# train = open(f"data/SBERT_test_topK_NoCons.csv", "w", encoding="utf-8")
	# train_w = csv.writer(train, delimiter='\t')
	"""只对主路径进行打分选择"""

	num = 0
	with open(f"data/webQSP/source_data/WebQSP.{path}.json", "r", encoding="utf-8") as f:
		# with open(f"../../Datasets/WQSP/test.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in tqdm(data["Questions"]):
			"""获取基本信息"""
			query = i["ProcessedQuestion"]
			mention = i["Parses"][0]["PotentialTopicEntityMention"]
			if mention is None:
				mention = ""
			query_e = query.replace(mention, "<e>")
			# query_e = query
			TopicEntityMid = i["Parses"][0]["TopicEntityMid"]
			InferentialChain = i["Parses"][0]["InferentialChain"]
			AnswerEntityMid = ""
			# 没有answer就跳过
			try:
				AnswerEntityMid = i["Parses"][0]["Answers"][0]["AnswerArgument"]
			except Exception as e:
				print(e)
				pass
			if AnswerEntityMid:
				answer_type = query_virtuoso_entityType(AnswerEntityMid)
				if answer_type:
					query_e = query_e + ' ' + '[' + answer_type + ']'
			"""生成训练数据"""
			# p_info = []
			if InferentialChain is not None:
				num += 1
				one_hop_path_scored = []
				one_hop_path_scored_sorted = []
				one_hop_path = get_1hop_p(TopicEntityMid)
				if one_hop_path:
					one_hop_path_scored = []
					for p in one_hop_path:
						if p != InferentialChain[0]:  # 剔除正样本
							one_path = "ns:" + p + ' ?x .'
							one_hop_path_scored.append([query_e, one_path, cosin_sim(query_e, p)])
					one_hop_path_scored_sorted = sorted(one_hop_path_scored, key=lambda x: x[2], reverse=True)
				if len(InferentialChain) == 1:
					train_w.writerow([query_e, "ns:" + InferentialChain[0] + " ?x .", 1])  # 写入主路径的正样本
					for j in one_hop_path_scored_sorted[:k]:
						train_w.writerow([j[0], j[1], 0])
				if len(InferentialChain) == 2:
					golden_path = f"ns:{InferentialChain[0]} ?y .?y ns:{InferentialChain[1]} ?x ."
					train_w.writerow([query_e, golden_path, 1])  # 写入主路径的正样本
					for ii in one_hop_path_scored_sorted[:k]:
						one_path_in_two = ii[1].split("ns:")[-1].replace(' ?x .', '')
						two_hop_path = get_2hop_p(TopicEntityMid, one_path_in_two)
						if two_hop_path:
							two_hop_path_scored = []
							for p2 in two_hop_path:
								two_path = f"ns:{one_path_in_two} ?y .?y ns:{p2} ?x ."
								if two_path != golden_path:  # 剔除正样本
									two_hop_path_scored.append([query_e, two_path, cosin_sim(query_e, p2)])
							two_hop_path_scored_sorted = sorted(two_hop_path_scored, key=lambda x: x[2], reverse=True)
							for j in two_hop_path_scored_sorted[:k]:
								train_w.writerow([j[0], j[1], 0])

		print(f"数量有{num}")


# 只生成测试集正确的样本
def only_true(path):
	train = open(f"data/webQSP/mask_data/{path}_onlytrue.csv", "w", encoding="utf-8", newline="")
	train_w = csv.writer(train, delimiter='\t')

	with open(f"data/webQSP/source_data/WebQSP.{path}.json", "r", encoding="utf-8") as f:
		# with open(f"../../Datasets/WQSP/test.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in data["Questions"]:
			"""获取基本信息"""
			query = i["ProcessedQuestion"]
			mention = i["Parses"][0]["PotentialTopicEntityMention"]
			# query_e = query.replace(mention, "<e>")
			query_e = query
			TopicEntityMid = i["Parses"][0]["TopicEntityMid"]
			# 没有answer就跳过
			try:
				AnswerEntityMid = i["Parses"][0]["Answers"][0]["AnswerArgument"]
			except Exception as e:
				print(e)
				continue
			InferentialChain = i["Parses"][0]["InferentialChain"]

			# '''添加生成约束的训练数据'''
			# Constraints = i["Parses"][0]["Constraints"]
			# print(Constraints)
			# cons = getConsPosPath(sparql)
			# if len(cons):
			#     for con in cons:
			#         print(query_e,con,1)
			#         num+=1
			#         train_w.writerow([num,query_e,con,1])  #不需要约束的负样本
			"""生成训练数据"""
			# p_info = []
			if InferentialChain is not None:
				train_w.writerow([query_e, InferentialChain[0], 1])  # 写入主路径的正样本
				# print(query_e, InferentialChain[0], 1)
				if len(InferentialChain) == 2:
					train_w.writerow([query_e, InferentialChain[1], 1])  # 写入主路径的正样本
		# print(query_e, InferentialChain[1], 1)


# 训练完后，用生成的权重pt文件预测测试集分数，计算打分最高的主路径是否为groud truth，并计算比例
def cal_true_path(path):
	# 测试集问题个数
	q_num = 1638
	# 排序第一是正确的路径个数
	true_num = 0
	num = 0
	with open(f"data/webQSP/source_data/WebQSP.{path}.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in tqdm(data["Questions"]):
			num += 1
			"""获取基本信息"""
			query = i["ProcessedQuestion"]
			mention = i["Parses"][0]["PotentialTopicEntityMention"]
			if mention is None:
				mention = ""
				continue
			# query_e = query.replace(mention, "<e>")
			TopicEntityMid = i["Parses"][0]["TopicEntityMid"]
			# 主题实体类型
			# top_e_typename = query_virtuoso_entityType(TopicEntityMid)
			# if top_e_typename:
			# 	query_e = query + " [unused0] " + top_e_typename
			# else:
			# 	query_e = query
			query_e = query
			# 没有answer就跳过
			try:
				Answer = i["Parses"][0]["Answers"][0]["AnswerArgument"]
			except Exception as e:
				print(e)
				continue
			InferentialChain = i["Parses"][0]["InferentialChain"]
			Scores_neg = []  # 负样本打分
			if InferentialChain is not None:
				one_hop_path = get_1hop_p(TopicEntityMid)
				if one_hop_path:
					for p in one_hop_path:
						score = cosin_sim(query_e, p)
						Scores_neg.append([p, score])
				else:
					print(f"问题{query_e}没有第一跳路径")
					continue
			# 倒序  选择Top_K负样本
			reverse_Score_neg = sorted(Scores_neg, key=lambda item: item[1], reverse=True)
			if reverse_Score_neg[0][0] not in InferentialChain:
				continue

			if len(InferentialChain) == 2:
				Scores_neg2 = []
				two_hop_path = get_2hop_p(TopicEntityMid, InferentialChain[0])
				if two_hop_path:
					for p2 in two_hop_path:
						if p2 not in InferentialChain:  # path不是正样本
							score2 = cosin_sim(query_e, p2)
							Scores_neg2.append([p2, score2])
				else:
					print(f"问题{query_e}没有第二跳路径")
					continue
				# 倒序  选择Top_K负样本
				reverse_Score_neg_2hop = sorted(Scores_neg2, key=lambda item: item[1], reverse=True)
				if reverse_Score_neg_2hop[0][0] not in InferentialChain:
					continue

			true_num += 1
			if num % 200 == 0:
				print(f"第{num}个问题，此时查正确的有{true_num}个")

		print(f"总共问题有{q_num},排序后正确的个数为{true_num},比例为{float(true_num / q_num)}")


# 生成complexQA的所有路径,注意这里原始数据的答案不是实体ID，而是实体名字
def create_complexQA_data_allpath(path):
	train = open(f"data/complexQA/mask_data/{path}_allpath_NoCons.csv", "w", encoding="utf-8", newline="")
	train_w = csv.writer(train, delimiter='\t')
	# train = open(f"data/SBERT_test_topK_NoCons.csv", "w", encoding="utf-8")
	# train_w = csv.writer(train, delimiter='\t')
	"""只对主路径进行打分选择"""

	num = 0

	with open(f"data/complexQA/source_data/compQ.{path}.json", "r", encoding="utf-8") as f:
		# with open(f"../../Datasets/WQSP/test.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in tqdm(data):
			"""获取基本信息"""
			query = i["question"]
			mention = i["mention"]
			if mention is None:
				mention = ""
			query_e = query.replace(mention, "<e>")
			# query_e = query
			TopicEntityMid = i["TopicEntityMid"]
			try:
				InferentialChain = i["InferentialChain"]
			except Exception as e:
				print(e)
				continue
			"""生成训练数据"""
			# p_info = []
			if InferentialChain is not None:
				num += 1
				if path == "test":
					train_w.writerow([query_e, InferentialChain[0], 1])  # 写入主路径的正样本
					print(query_e, InferentialChain[0], 1)
					one_hop_path = get_1hop_p(TopicEntityMid)
					if one_hop_path:
						for p in one_hop_path:
							if p not in InferentialChain:  # 剔除正样本
								train_w.writerow([query_e, p, 0])
								print(query_e, p, 0)
					if len(InferentialChain) == 2:
						train_w.writerow([query_e, InferentialChain[1], 1])  # 写入主路径的正样本
						print(query_e, InferentialChain[1], 1)
						two_hop_path = get_2hop_p(TopicEntityMid, InferentialChain[0])
						if two_hop_path:
							for p2 in two_hop_path:
								if p2 not in InferentialChain:  # 剔除正样本
									train_w.writerow([query_e, p2, 0])
									print(query_e, p2, 0)
				elif path == "train":
					train_w.writerow([query_e, InferentialChain[0], 1])  # 写入主路径的正样本
					print(query_e, InferentialChain[0], 1)
					one_hop_path = get_1hop_p(TopicEntityMid)
					if one_hop_path:
						for p in one_hop_path:
							if p not in InferentialChain:  # 剔除正样本
								# core_path_neg = p + ' [unused0] ' + top_e_typename
								train_w.writerow([query_e, p, 0])
								print(query_e, p, 0)
					if len(InferentialChain) == 2:
						# core_path_pos = InferentialChain[1] + ' [unused0] ' + top_e_typename
						train_w.writerow([query_e, InferentialChain[1], 1])  # 写入主路径的正样本
						print(query_e, InferentialChain[1], 1)
						two_hop_path = get_2hop_p(TopicEntityMid, InferentialChain[0])
						if two_hop_path:
							for p2 in two_hop_path:
								if p2 not in InferentialChain:  # 剔除正样本
									# core_path_neg = p2 + ' [unused0] ' + top_e_typename
									train_w.writerow([query_e, p2, 0])
									print(query_e, p2, 0)
		print(f"数量有{num}")

def create_complexQA_data_beam_search_top_k(path, k):
	train = open(f"data/complexQA/beam_search/{path}_top_k_NoCons.csv", "w", encoding="utf-8", newline="")
	train_w = csv.writer(train, delimiter='\t')
	# train = open(f"data/SBERT_test_topK_NoCons.csv", "w", encoding="utf-8")
	# train_w = csv.writer(train, delimiter='\t')
	"""只对主路径进行打分选择"""

	num = 0

	with open(f"data/complexQA/source_data/compQ.{path}.json", "r", encoding="utf-8") as f:
		# with open(f"../../Datasets/WQSP/test.json", "r", encoding="utf-8") as f:
		data = json.load(f)
		for i in tqdm(data):
			"""获取基本信息"""
			query = i["question"]
			mention = i["mention"]
			if mention is None:
				mention = ""
			query_e = query.replace(mention, "<e>")
			# query_e = query
			TopicEntityMid = i["TopicEntityMid"]
			try:
				InferentialChain = i["InferentialChain"]
			except Exception as e:
				print(e)
				continue
			"""生成训练数据"""
			# p_info = []
			if InferentialChain is not None:
				num += 1
				one_hop_path_scored_sorted = []
				one_hop_path = get_1hop_p(TopicEntityMid)
				if one_hop_path:
					one_hop_path_scored = []
					for p in one_hop_path:
						if p != InferentialChain[0]:  # 剔除正样本
							one_path = "ns:" + p + ' ?x .'
							one_hop_path_scored.append([query_e, one_path, cosin_sim(query_e, p)])
					one_hop_path_scored_sorted = sorted(one_hop_path_scored, key=lambda x: x[2], reverse=True)
				if len(InferentialChain) == 1:
					train_w.writerow([query_e, "ns:" + InferentialChain[0] + " ?x .", 1])  # 写入主路径的正样本
					for j in one_hop_path_scored_sorted[:k]:
						train_w.writerow([j[0], j[1], 0])
				if len(InferentialChain) == 2:
					golden_path = f"ns:{InferentialChain[0]} ?y .?y ns:{InferentialChain[1]} ?x ."
					train_w.writerow([query_e, golden_path, 1])  # 写入主路径的正样本
					for ii in one_hop_path_scored_sorted[:k]:
						one_path_in_two = ii[1].split("ns:")[-1].replace(' ?x .', '')
						two_hop_path = get_2hop_p(TopicEntityMid, one_path_in_two)
						if two_hop_path:
							two_hop_path_scored = []
							for p2 in two_hop_path:
								two_path = f"ns:{one_path_in_two} ?y .?y ns:{p2} ?x ."
								if two_path != golden_path:  # 剔除正样本
									two_hop_path_scored.append([query_e, two_path, cosin_sim(query_e, p2)])
							two_hop_path_scored_sorted = sorted(two_hop_path_scored, key=lambda x: x[2], reverse=True)
							for j in two_hop_path_scored_sorted[:k]:
								train_w.writerow([j[0], j[1], 0])
		print(f"数量有{num}")


# 生成问题的id与主题实体名字、id和类型的映射关系
def create_qid_toptype_topid(path):
	data_path = f"data/webQSP/source_data/WebQSP.{path}.json"
	out_path = f"data/webQSP/{path}_qid_top_Ename_Eid_Etype.json"
	with open(data_path, "r", encoding="utf-8") as f:
		output = {}
		data = json.load(f)
		for i in data["Questions"]:
			topicname = i["Parses"][0]["TopicEntityName"]
			topicEntityID = i["Parses"][0]["TopicEntityMid"]
			typename = query_virtuoso_entityType(topicEntityID)
			output[i["QuestionId"]] = {"topicEntityName": topicname, "topicEntityID": topicEntityID, "topicEntityTypeName": typename}
		with open(out_path, 'w', encoding="utf-8") as f1:
			json.dump(output, f1)

# 生成候选主题实体约束
def get_sim_entitycons(name):

	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
	out_path = f"data/webQSP/candid_cons/{name}_entitycons_label.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		all = {}
		for i in tqdm(data["Questions"]):
			QuestionId = i["QuestionId"]
			# if int(q_id.replace("WebQTest-", "")) > -1 :
			# if int(q_id.replace("WebQTrn-", "")) > -1:

			# 存储问题的基本信息 id 问题 提及 实体id
			if i["Parses"][0]["PotentialTopicEntityMention"] and i["Parses"][0]["TopicEntityMid"] and i["Parses"][0][
				"TopicEntityName"] and i["Parses"][0]["Sparql"]:  # 一定数据要都存在再处理
				question = i["ProcessedQuestion"].lower()  # 大写转小写
				mention = i["Parses"][0]["PotentialTopicEntityMention"].lower()
				topicEntityID = i["Parses"][0]["TopicEntityMid"]
				topicEntityName = i["Parses"][0]["TopicEntityName"]
				cons = i["Parses"][0]["Constraints"]
				# 存放两跳
				cand_list = []
				# 一跳
				cand_list1 = {}
				cand_path_label1 = {}
				# 二跳
				cand_list2 = {}
				cand_path_label2 = {}
				gold_path_list = []
				is_cons = False
				if i["Parses"][0]["InferentialChain"] and cons:
					PosQueryGraph = i["Parses"][0]["InferentialChain"]
				else:
					continue
				for i in cons:
					if i["ArgumentType"] == "Entity":
						if i["SourceNodeIndex"] == 0:
							cand_path_label1[i["NodePredicate"]] = 1
						else:
							cand_path_label2[i["NodePredicate"]] = 1
						gold_path_list.append(i["NodePredicate"])
						is_cons = True
				if not is_cons:
					continue

				# 返回字符串列表
				candid_entitycons_path = get_candid_entitycons(question, topicEntityID, PosQueryGraph[0], False)
				if candid_entitycons_path:
					for i in candid_entitycons_path:
						if i in gold_path_list:
							continue
						cand_path_label1[i] = 0
				cand_list1[PosQueryGraph[0]] = cand_path_label1
				cand_list.append(cand_list1)
				if len(PosQueryGraph) == 2:
					candid_entitycons_path = get_candid_entitycons(question, topicEntityID, f"ns:{PosQueryGraph[0]} ?y . ?y ns:{PosQueryGraph[1]} ?x .", True)
					if candid_entitycons_path:
						for i in candid_entitycons_path:
							if i in gold_path_list:
								continue
							cand_path_label2[i] = 0
					cand_list2[PosQueryGraph[1]] = cand_path_label2
					cand_list.append(cand_list2)
				all[QuestionId] = cand_list
				print(cand_list)
		with open(out_path, 'w', encoding="utf-8") as f1:
			json.dump(all, f1)

# 把提取出来的候选实体约束文件处理成模型训练的数据格式
def get_entitycons_train_test_data(path):
	data_path = f"data/webQSP/candid_cons/{path}_entitycons_label.json"
	f1 = open("data/webQSP/candid_cons/train.csv", 'w', newline="")
	f2 = open("data/webQSP/candid_cons/dev.csv", 'w', newline="")
	f3 = open("data/webQSP/candid_cons/test.csv", 'w', newline="")
	f1_writer = csv.writer(f1, delimiter='\t')
	f2_writer = csv.writer(f2, delimiter='\t')
	f3_writer = csv.writer(f3, delimiter='\t')
	f1_writer.writerow(["sent0", "sent1", "hard_neg"])
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 生成训练集
		for key, val in tqdm(list(data.items())[:int(len(data) * 0.8)]):
			for i in val:
				for core_path, cand_cons in i.items():
					# 如果第一个值不为1则该结点无实体约束
					if cand_cons:
						if cand_cons[next(iter(cand_cons))] == 0:
							continue
					else:
						continue
					true_path = ""
					for path, score in cand_cons.items():
						if score == 1:
							true_path = path
							continue
						f1_writer.writerow([core_path, true_path, path])
		# 生成测试/验证集
		for key, val in tqdm(list(data.items())[int(len(data) * 0.8):int(len(data) * 0.9)]):
			for i in val:
				for core_path, cand_cons in i.items():
					for path, score in cand_cons.items():
						f2_writer.writerow([core_path, path, score])
		for key, val in tqdm(list(data.items())[int(len(data) * 0.9):]):
			for i in val:
				for core_path, cand_cons in i.items():
					for path, score in cand_cons.items():
						f3_writer.writerow([core_path, path, score])
	f1.close()
	f2.close()
	f3.close()



# 生成core path和候选约束路径的映射关系
def get_cons_path(name):

	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
	out_path = f"data/webQSP/candid_cons/{name}_cons_path_label.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		all = {}
		for i in tqdm(data["Questions"]):
			QuestionId = i["QuestionId"]
			# if int(q_id.replace("WebQTest-", "")) > -1 :
			# if int(q_id.replace("WebQTrn-", "")) > -1:

			# 存储问题的基本信息 id 问题 提及 实体id
			if i["Parses"][0]["PotentialTopicEntityMention"] and i["Parses"][0]["TopicEntityMid"] and i["Parses"][0][
				"TopicEntityName"] and i["Parses"][0]["Sparql"]:  # 一定数据要都存在再处理
				question = i["ProcessedQuestion"].lower()  # 大写转小写
				mention = i["Parses"][0]["PotentialTopicEntityMention"].lower()
				topicEntityID = i["Parses"][0]["TopicEntityMid"]
				query_e = question.replace(mention, "<e>")
				# topicEntityName = i["Parses"][0]["TopicEntityName"]
				cons = i["Parses"][0]["Constraints"]
				order = i["Parses"][0]["Order"]
				# 查询是否有加在第二个节点的约束
				is_con_two = False
				# 查询是否有加在第一个节点的约束
				is_con_one = False
				# 存放一个问题的所有跳的约束路径
				cand_list = []
				# 一跳
				cand_list1 = {}
				cand_path_label1 = {}
				# 二跳
				cand_list2 = {}
				cand_path_label2 = {}
				gold_path_list = []
				is_cons = False
				# 排序约束和其他约束总得有一个
				if i["Parses"][0]["InferentialChain"] and (cons or order):
					PosQueryGraph = i["Parses"][0]["InferentialChain"]
				else:
					continue
				# 构造约束正样本,这里包含时间，实体等约束路径，排序需要单独处理
				if cons:
					for i in cons:
						if i["SourceNodeIndex"] == 0:
							cand_path_label1[i["NodePredicate"]] = 1
							is_con_one = True
						else:
							cand_path_label2[i["NodePredicate"]] = 1
							is_con_two = True
						gold_path_list.append(i["NodePredicate"])
				# 或者有排序约束
				if order:
					if order["SourceNodeIndex"] == 0:
						cand_path_label1[order["NodePredicate"]] = 1
						is_con_one = True
					else:
						cand_path_label2[order["NodePredicate"]] = 1
						is_con_two = True
					gold_path_list.append(order["NodePredicate"])

				# 返回字符串列表,一跳问题
				if len(PosQueryGraph) == 1 and is_con_one:
					candid_con_paths = get_candid_con_path(question, topicEntityID, PosQueryGraph[0], False)
					if candid_con_paths:
						for i in candid_con_paths:
							if i in gold_path_list:
								continue
							cand_path_label1[i] = 0
					cand_list1[query_e + ' ' + PosQueryGraph[0]] = cand_path_label1
					cand_list.append(cand_list1)
				elif len(PosQueryGraph) == 2:
					merged_words = PosQueryGraph[0].split(".") + PosQueryGraph[1].split(".")
					merge_path = ".".join(list(OrderedDict.fromkeys(merged_words)))
					if is_con_one:
						candid_con_paths = get_candid_con_path(question, topicEntityID, f"ns:{PosQueryGraph[0]} ?y .?y ns:{PosQueryGraph[1]} ?x .", True)
						if candid_con_paths:
							for i in candid_con_paths:
								if i in gold_path_list:
									continue
								cand_path_label1[i] = 0
						cand_list1[query_e + ' ' + merge_path] = cand_path_label1
						cand_list.append(cand_list1)
					if is_con_two:
						candid_con_paths = get_candid_con_path(question, topicEntityID, f"ns:{PosQueryGraph[0]} ?y .?y ns:{PosQueryGraph[1]} ?x .", True, True)
						if candid_con_paths:
							for i in candid_con_paths:
								if i in gold_path_list:
									continue
								cand_path_label2[i] = 0
						cand_list2[query_e + ' ' + merge_path] = cand_path_label2
						cand_list.append(cand_list2)
				else:
					continue

				all[QuestionId] = cand_list
				print(cand_list)
		with open(out_path, 'w', encoding="utf-8") as f1:
			json.dump(all, f1)

# 把提取出来的候选约束路径文件处理成模型训练的数据格式
def get_con_path_train_test_data(path):
	data_path = f"data/webQSP/candid_cons/{path}_cons_path_label.json"
	f1 = open("data/webQSP/candid_cons/train_cons_path.csv", 'w', newline="")
	f2 = open("data/webQSP/candid_cons/dev_cons_path.csv", 'w', newline="")
	f3 = open("data/webQSP/candid_cons/test_cons_path.csv", 'w', newline="")
	f1_writer = csv.writer(f1, delimiter='\t')
	f2_writer = csv.writer(f2, delimiter='\t')
	f3_writer = csv.writer(f3, delimiter='\t')
	f1_writer.writerow(["sent0", "sent1", "hard_neg"])
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 生成训练集
		for key, val in tqdm(list(data.items())[:int(len(data) * 0.8)]):
			for i in val:
				# q_core_path 问题和主路径的拼接
				for q_core_path, cand_cons in i.items():
					# 如果第一个值不为1则该结点无实体约束
					if cand_cons:
						if cand_cons[next(iter(cand_cons))] == 0:
							continue
					else:
						continue
					# 将正样本提取出来
					true_path = []
					for path, score in cand_cons.items():
						if score == 1:
							true_path.append(path)
							continue
					for t in true_path:
						for path, score in cand_cons.items():
							if score == 1:
								continue
							f1_writer.writerow([q_core_path, t, path])
		# 生成测试/验证集
		for key, val in tqdm(list(data.items())[int(len(data) * 0.8):int(len(data) * 0.9)]):
			for i in val:
				for q_core_path, cand_cons in i.items():
					for path, score in cand_cons.items():
						f2_writer.writerow([q_core_path, path, score])
		for key, val in tqdm(list(data.items())[int(len(data) * 0.9):]):
			for i in val:
				for q_core_path, cand_cons in i.items():
					for path, score in cand_cons.items():
						f3_writer.writerow([q_core_path, path, score])
	f1.close()
	f2.close()
	f3.close()




# 从原始数据获取问题跳数分类数据
def hop_classification(train_path, dev_path, test_path):
	data_path1 = f"data/complexQA/source_data/compQ.train.json"
	data_path2 = f"data/complexQA/source_data/compQ.test.json"
	# data_path1 = f"data/webQSP/source_data/WebQSP.train.json"
	# data_path2 = f"data/webQSP/source_data/WebQSP.test.json"
	f1 = open(data_path1, "r", encoding="utf-8")
	f2 = open(data_path2, "r", encoding="utf-8")
	ftrain = open(train_path, 'a', encoding='utf-8', newline="")
	fdev = open(dev_path, 'a', encoding='utf-8', newline="")
	ftest = open(test_path, 'a', encoding='utf-8', newline="")
	data1 = json.load(f1)
	# data1 = data1["Questions"]
	for i in tqdm(data1[:int(len(data1) * 0.8)]):
		# question = i["ProcessedQuestion"].lower()  # 大写转小写
		question = i["question"].lower()  # 大写转小写
		if question[-1] == '?':
			question = question[:-1]
		# if i["Parses"][0]["InferentialChain"]:
		try:
			InferentialChain = i["InferentialChain"]
		except Exception as e:
			print(e)
			continue
		if InferentialChain:
			# hop = len(i["Parses"][0]["InferentialChain"]) - 1
			hop = len(i["InferentialChain"]) - 1
		else: continue
		ftrain.write(question + '\t' + str(hop) + '\n')
	for i in tqdm(data1[int(len(data1) * 0.8):]):
		# question = i["ProcessedQuestion"].lower()  # 大写转小写
		question = i["question"].lower()  # 大写转小写
		if question[-1] == '?':
			question = question[:-1]
		# if i["Parses"][0]["InferentialChain"]:
		try:
			InferentialChain = i["InferentialChain"]
		except Exception as e:
			print(e)
			continue
		if InferentialChain:
			# hop = len(i["Parses"][0]["InferentialChain"]) - 1
			hop = len(i["InferentialChain"]) - 1
		else: continue
		fdev.write(question + '\t' + str(hop) + '\n')
	data2 = json.load(f2)
	# data2 = data2["Questions"]
	for i in tqdm(data2):
		# question = i["ProcessedQuestion"].lower()  # 大写转小写
		question = i["question"].lower()  # 大写转小写
		if question[-1] == '?':
			question = question[:-1]
		# if i["Parses"][0]["InferentialChain"]:
		try:
			InferentialChain = i["InferentialChain"]
		except Exception as e:
			print(e)
			continue
		if InferentialChain:
			# hop = len(i["Parses"][0]["InferentialChain"]) - 1
			hop = len(i["InferentialChain"]) - 1
		else: continue
		ftest.write(question + '\t' + str(hop) + '\n')

# 对比学习问题分类数据构建
def create_hop_classi_data():
	# 将一跳和两跳的问题进行分类并写入文件
	# f = open("data/webQSP/hop_classification/contras_data/all.csv", "r", encoding="utf-8")
	# f1 = open("data/webQSP/hop_classification/one_hop.csv", 'w', encoding="utf-8", newline='')
	# f2 = open("data/webQSP/hop_classification/two_hop.csv", 'w', encoding="utf-8", newline='')
	# reader = csv.reader(f, delimiter='\t')
	# writer1 = csv.writer(f1, delimiter='\t')
	# writer2 = csv.writer(f2, delimiter='\t')
	# one_num = 0
	# two_num = 0
	# for i in reader:
	# 	if i[1] == '0':
	# 		one_num += 1
	# 		writer1.writerow(i)
	# 	else:
	# 		two_num += 1
	# 		writer2.writerow(i)
	# print(one_num, two_num)

	# 构建训练、验证、测试集
	f1 = open("data/webQSP/hop_classification/one_hop.csv", 'r', encoding="utf-8")
	f2 = open("data/webQSP/hop_classification/two_hop.csv", 'r', encoding="utf-8")
	# ftrain = open("data/webQSP/hop_classification/contras_data/train.csv", 'w', encoding='utf-8', newline="")
	fdev = open("data/webQSP/hop_classification/contras_data/dev.csv", 'w', encoding='utf-8', newline="")
	ftest = open("data/webQSP/hop_classification/contras_data/test.csv", 'w', encoding='utf-8', newline="")
	# writer_train = csv.writer(ftrain, delimiter='\t')
	writer_dev = csv.writer(fdev, delimiter='\t')
	writer_test = csv.writer(ftest, delimiter='\t')
	reader1 = csv.reader(f1, delimiter='\t')
	reader2 = csv.reader(f2, delimiter='\t')
	train_data = []
	train_data2 = []
	dev_data = []
	test_data = []
	temp_data = []
	num = 0
	# # 构造训练集
	# for i in reader1:
	# 	num += 1
	# 	temp_data.append(i[0])
	# 	if num == 2:
	# 		num = 0
	# 		train_data.append(temp_data)
	# 		temp_data = []
	# for i, j in zip(train_data, reader2):
	# 	i.append(j[0])
	# 	train_data2.append(i)
	# train_data = []
	# temp_data = []
	# f1.seek(0)
	# f2.seek(0)
	# reader2 = csv.reader(f2, delimiter='\t')
	# reader1 = csv.reader(f1, delimiter='\t')
	# for i in reader2:
	# 	num += 1
	# 	temp_data.append(i[0])
	# 	if num == 2:
	# 		num = 0
	# 		train_data.append(temp_data)
	# 		temp_data = []
	# for i, j in zip(train_data, reader1):
	# 	i.append(j[0])
	# 	train_data2.append(i)
	# random.shuffle(train_data2)
	# writer_train.writerows(train_data2)

	# 构造测试,验证集

	temp_data = []
	num = 0
	lines1 = list(reader1)
	lines2 = list(reader2)
	for i, j in zip(lines1[int(len(lines1) * 0.8):int(len(lines1) * 0.9)], lines2):
		dev_data.append([i[0], j[0], 0])
	for i, j in zip(lines1[int(len(lines1) * 0.9):], lines2[int(len(lines2) * 0.6):]):
		test_data.append([i[0], j[0], 0])
	for i in lines1[:int(len(lines1) * 0.1) * 2]:
		num += 1
		temp_data.append(i[0])
		if num == 2:
			num = 0
			temp_data.append(1)
			dev_data.append(temp_data)
			temp_data = []
	for i in lines2[:int(len(lines2) * 0.1) * 2]:
		num += 1
		temp_data.append(i[0])
		if num == 2:
			num = 0
			temp_data.append(1)
			test_data.append(temp_data)
			temp_data = []

	random.shuffle(test_data)
	random.shuffle(dev_data)
	writer_dev.writerows(dev_data)
	writer_test.writerows(test_data)



if __name__ == '__main__':
	generate_train_data("data/complexQA/beam_search/train_top_k_NoCons.csv", "data/complexQA/beam_search/train.csv")
	# create_qid_toptype_topid("test")
	# get_sim_entitycons("train")
	# get_entitycons_train_test_data("train")
	# create_complexQA_data_allpath("test")
	# generate_train_data("data/complexQA/mask_data/train_allpath.csv", "data/complexQA/mask_data/train_allpath.csv")
	# create_hop_classi_data()
	# hop_classification("data/webQSP/hop_classification/question_label/train.txt", "data/webQSP/hop_classification/question_label/dev.txt", "data/webQSP/hop_classification/question_label/test.txt")
	# create_hop_classi_data()
	# get_cons_path("test")
	# get_con_path_train_test_data("test")
	# create_data_beam_search_top_k("test", 5)
	# create_complexQA_data_beam_search_top_k("test", 5)