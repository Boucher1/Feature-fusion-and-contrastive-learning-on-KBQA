#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/14 18:56
# @Author  : Xavier Byrant
# @FileName: data_process.py
# @Software: PyCharm
import json

from tqdm import tqdm

from F1_test import hit_n

# # 去掉第一列索引(文件夹下所有csv文件)
# import csv
# import os
# folder_path = './data/webQSP/'
#
# # 获取指定文件夹下的所有文件名
# filenames = os.listdir(folder_path)
#
# # 循环遍历所有文件，筛选出符合条件的 TSV 文件并输出其文件名
# for filename in filenames:
#     if filename.endswith('.csv'):
#         # Read TSV file and remove first column
#         file_path = os.path.join(folder_path, filename)
#
#         with open(file_path, 'r') as file:
#             reader = csv.reader(file, delimiter='\t')
#             rows = []
#             for row in reader:
#                 rows.append(row[1:])
#             # Write updated data back to TSV file
#             with open(file_path, 'w', newline='') as file:
#                 writer = csv.writer(file, delimiter='\t')
#                 for row in rows:
#                     writer.writerow(row)


# # 去掉第一列索引(单个文件)
# import csv
#
# file_path = "data/webQSP/nomask_data/SBERT_test_NoCons_nomask_allpath.csv"
# with open(file_path, 'r') as file:
# 	reader = csv.reader(file, delimiter='\t')
# 	rows = []
# 	for row in reader:
# 		rows.append(row[1:])
# 	# Write updated data back to TSV file
# 	with open(file_path, 'w', newline='') as file:
# 		writer = csv.writer(file, delimiter='\t')
# 		writer.writerow(['sent0', 'sent1', 'label'])
# 		writer.writerows(rows)

# # tsv转csv
# import csv
# import os
# folder_path = './data/complexQA/'
#
# # 获取指定文件夹下的所有文件名
# filenames = os.listdir(folder_path)
#
# # 循环遍历所有文件，筛选出符合条件的 TSV 文件并输出其文件名
# for filename in filenames:
# 	if filename.endswith('.tsv'):
# 		file_path = os.path.join(folder_path, filename)
# 		print(file_path)
# 		with open(file_path, 'r') as file:
# 			reader = csv.reader(file, delimiter='\t')
# 			rows = []
# 			for row in reader:
# 				rows.append(row)
# 			with open(file_path[:-4] + ".csv", 'w', newline='') as file:
# 				writer = csv.writer(file, delimiter='\t')
# 				writer.writerows(rows)

# # tsv转csv
# import pandas as pd
# import os
#
# folder_path = './data/webQSP/'
#
# # 获取指定文件夹下的所有文件名
# file_names = os.listdir(folder_path)
#
# # 循环遍历所有文件，筛选出符合条件的 TSV 文件并输出其文件名
# for file_name in file_names:
#     if file_name.endswith('.tsv'):
#         # 构造输入和输出文件的路径
#         input_file_path = os.path.join(folder_path, file_name)
#         output_file_path = os.path.join(folder_path, file_name.replace('.tsv', '.csv'))
#         # 读取 TSV 文件并保存为 CSV 文件
#         df = pd.read_csv(input_file_path, sep='\t')
#         df.to_csv(output_file_path, index=False,sep='\t')


# # 打乱顺序
# import random
#
# # 读取txt文件内容
# with open("data/webQSP/nomask_data/valid_nomask_oversample.csv", "r") as f:
# 	lines = f.readlines()
# 	# lines = lines[1:]
# 	# 打乱行顺序
# 	random.shuffle(lines)
# 	# 将打乱后的内容写回文件
# 	with open("data/webQSP/nomask_data/valid_nomask_oversampleshuffle.csv", "w") as f:
# 		# f.write("sent0	sent1	label"+'\n')
# 		f.writelines(lines)


# # 验证集和训练集划分
# with open("./data/complexQA/mask_data/train_valid_allpath_NoCons.csv", "r") as f:
# 	lines = f.readlines()
# 	# lines = lines[1:]
# 	# 删除第一列
# 	# lines = [line.split('\t', 1)[1] for line in lines]
# 	l = len(lines)
# 	with open("./data/complexQA/mask_data/train_allpath.csv", 'w') as f1:
# 		f1.write("sent0	sent1	label"+'\n')
# 		for line in lines[0:int(l * 0.8)]:
# 			f1.write(line)
# 	with open("./data/complexQA/mask_data/valid_allpath.csv", 'w') as f1:
# 		# f1.write("sent0	sent1	label"+'\n')
# 		for line in lines[int(l * 0.8):]:
# 			f1.write(line)

# # 去掉type
# import csv
#
# f1 = open("data/webQSP/nomask_data/valid_nomask_oversample.csv", 'w', newline='')
# with open("data/webQSP/nomask_data/valid_nomask_e_type_oversample.csv", 'r') as f2:
# 	reader = csv.reader(f2, delimiter='\t')
# 	writer = csv.writer(f1, delimiter='\t')
# 	for line in reader:
# 		writer.writerow([line[0].split(" [unused0] ")[0], line[1].split(" [unused0] ")[0], line[2]])


# # 去掉[unused0]
# import csv
#
# f1 = open("data/webQSP/nomask_data/train_nomask_e_nounused0_type.csv", 'w', newline='')
# with open("data/webQSP/nomask_data/train_nomask_e_type.csv", 'r') as f2:
# 	reader = csv.reader(f2, delimiter='\t')
# 	next(reader)
# 	writer = csv.writer(f1, delimiter='\t')
# 	writer.writerow(['sent0', 'sent1', 'hard_neg'])
# 	for line in reader:
# 		writer.writerow([line[0].replace("[unused0] ", ''), line[1].replace("[unused0] ", ''), line[2].replace("[unused0] ", '')])


# 提取webquestionsp带约束和不带约束的问题并按跳数分类
data_path = "data/webQSP/source_data/WebQSP.test.json"
cons_one_hop_data = {}
cons_two_hop_data = {}
uncons_one_hop_data = {}
uncons_two_hop_data = {}

cons_one_hop_Questions = []
cons_two_hop_Questions = []
uncons_one_hop_Questions = []
uncons_two_hop_Questions = []

with open(data_path, "r", encoding="utf-8") as f:
	data = json.load(f)
	for i in tqdm(data["Questions"]):
		if i["Parses"][0]["PotentialTopicEntityMention"] and i["Parses"][0]["TopicEntityMid"] and i["Parses"][0][
			"TopicEntityName"] and i["Parses"][0]["Sparql"]:
			if i["Parses"][0]["Constraints"]:
				if i["Parses"][0]["InferentialChain"]:
					if len(i["Parses"][0]["InferentialChain"]) == 1:
						cons_one_hop_Questions.append(i)
					else:
						cons_two_hop_Questions.append(i)
			else:
				if i["Parses"][0]["InferentialChain"]:
					if len(i["Parses"][0]["InferentialChain"]) == 1:
						uncons_one_hop_Questions.append(i)
					else:
						uncons_two_hop_Questions.append(i)
	cons_one_hop_data["Questions"] = cons_one_hop_Questions
	cons_two_hop_data["Questions"] = cons_two_hop_Questions
	uncons_one_hop_data["Questions"] = uncons_one_hop_Questions
	uncons_two_hop_data["Questions"] = uncons_two_hop_Questions
	f1 = open("data/webQSP/source_data/WebQSP.test.cons_one_hop.json", 'w', encoding="utf-8")
	f2 = open("data/webQSP/source_data/WebQSP.test.cons_two_hop.json", 'w', encoding="utf-8")
	f3 = open("data/webQSP/source_data/WebQSP.test.uncons_one_hop.json", 'w', encoding="utf-8")
	f4 = open("data/webQSP/source_data/WebQSP.test.uncons_two_hop.json", 'w', encoding="utf-8")
	json.dump(cons_one_hop_data, f1)
	json.dump(cons_two_hop_data, f2)
	json.dump(uncons_one_hop_data, f3)
	json.dump(uncons_two_hop_data, f4)
		# print(hit_n(["a", "b", "c", "d", "e"], ["a", "f", "g", "h", "i"], 3))