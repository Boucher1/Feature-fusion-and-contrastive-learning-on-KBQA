#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/27
# @Author  : hehl
# @Software: PyCharm
# @File    : query_virtuoso.py
import nltk
import warnings

warnings.filterwarnings('ignore',
						message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead")

from SPARQLWrapper import SPARQLWrapper, JSON
from utils import cosin_sim
from cons_select import cons_cosin_sim
import eventlet  # 导入eventlet这个模块

import re

"""链接virtuoso 设置返回数据为json格式"""
# sparqlQuery = SPARQLWrapper("http://43.142.132.166:8890/sparql")
# model_path = 'D:/workspace/pythonProject/complexQA/dyq_complexQA/queryGraph/SBERT/data/preTrain'

sparqlQuery = SPARQLWrapper("http://172.23.253.79:8890/sparql")
# model_path = "../SBERT/data/preTrainall"
# model_path = "../SBERT/data/CQpreTrainall"

sparqlQuery.setReturnFormat(JSON)

eventlet.monkey_patch()  # 必须加这条代码

import time

"""微调bert  计算相似度"""


# model = SentenceTransformer(model_path)
# print(model)

# def sbert_score(s1, s2):
#     embedding1 = model.encode(s1, convert_to_tensor=True)
#     embedding2 = model.encode(s2, convert_to_tensor=True)
#
#     similarity = util.cos_sim(embedding1, embedding2)
#
#     vecdict = dict(zip(s2, similarity.cpu().numpy().tolist()[0]))
#     # vecdict = dict(zip(s2, similarity.numpy().tolist()[0]))
#
#     return sorted(vecdict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
#


def replace_rdf(p):
	return p.replace("http://rdf.freebase.com/ns/", "")


def is_query_anser(entityID, queryGraph):
	if queryGraph.count("{") == queryGraph.count("}"):
		queryGraph = queryGraph + "}"
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT  ?x WHERE { FILTER (?x != ns:" + entityID + ") ns:" + entityID + " " + queryGraph
	# print(sparql)
	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			if len(results):
				return True
			return False
		except Exception as e:
			print("发生错误:", e)
			time.sleep(5)
			continue
	return False


# 查询一跳路径
def get_1hop_p(entityID):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select DISTINCT ?p  where{ ns:%s  ?p ?o}" % (entityID)
		# print(query_txt)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						temp = replace_rdf(results[i]["p"]["value"])
						if temp not in data:
							data.append(temp)
					# 返回不重复的Path
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 查询两跳路径
def get_2hop_p(entityID, one_hop_path):
	if "m." in entityID or "g." in entityID:
		while True:
			try:
				query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p  where{ ns:%s ns:%s ?x.?x ?p ?o}" % (
					entityID, one_hop_path)
				# print(query_txt)
				with eventlet.Timeout(60, False):
					sparqlQuery.setQuery(query_txt)
					results = sparqlQuery.query().convert()["results"]["bindings"]
					data = []
					if len(results):
						for i in range(len(results)):
							p = replace_rdf(results[i]["p"]["value"])
							if p not in data:
								data.append(p)
						# 返回不重复的Path
						# print(data)
						return data
					else:
						return None
				return None
			except Exception as e:
				print("Exception:", e)
				time.sleep(5)
				continue  # 间隔5秒以便重试


# 查找问题中的年份约束
def find_year(text):
	pattern = r'\b\d{4}\b'  # 匹配四位数的数字
	year_list = []
	for match in re.finditer(pattern, text):
		year = int(match.group())  # 遍历匹配结果，转换成整数后添加到列表中
		if year >= 1000 and year <= 9999:  # 判断是否为年份
			year_list.append(year)
	return year_list


# 查询一跳带*.*.from和*.*.to的时间约束的路径
"""一跳的时间约束"""


def query_1hop_p_from_to(entityID, one_hop_path, yearData):
	# sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>  SELECT DISTINCT ?p ?o  WHERE { ns:%s %s ?y . ?y ?p ?o. FILTER( CONTAINS(STR(?p), \".from\") || CONTAINS(STR(?p), \".to\") )}" % (entityID, one_hop_path)  # 仅仅查找出包含 from 或者t o的路径
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>  SELECT DISTINCT ?p ?o  WHERE { ns:%s %s ?x ?p ?o. FILTER(CONTAINS(STRAFTER(STRAFTER(STRAFTER(STRAFTER(STR(?p), \".\"), \".\"), \".\"), \".\"), \".from\") || CONTAINS(STRAFTER(STRAFTER(STRAFTER(STRAFTER(STR(?p), \".\"), \".\"), \".\"), \".\"), \".to\") || CONTAINS(STRAFTER(STRAFTER(STRAFTER(STRAFTER(STR(?p), \".\"), \".\"), \".\"), \".\"), \"date\"))}" % (
		entityID, one_hop_path)  # 仅仅查找出包含 from 或者t o的路径
	print(sparql)

	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			print(results)
			from_path = ""
			to_path = ""
			if len(results):
				for i in range(len(results)):
					# 是否考虑多个from to的路径？
					p = replace_rdf(results[i]["p"]["value"])
					o = replace_rdf(results[i]["o"]["value"])
					if str(yearData) in o and ".from" in p:
						from_path = p
					if str(yearData) in o and ".to" in p:
						to_path = p
					if str(yearData) in o and "date" in p:
						# 针对"from_date"的路径
						if "from" in p:
							from_path = p
						elif "to" in p:
							to_path = p
						else:
							from_path = p
							to_path = p
				if from_path and to_path:
					return from_path, to_path
				return None, None
			else:
				return None, None
		except Exception as e:
			print("发生错误:", e)
			time.sleep(5)
			continue


# 查询一跳的时间约束并升序排列
def query_order_asc_1hop(entityID, one_hop_path):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p  ?n where{ ns:%s %s ?x  ?p ?n.FILTER( CONTAINS(STR(?p), \".from\") || CONTAINS(STR(?p), \"_date\") )}" % (
		entityID, one_hop_path)
	# print(sparql)

	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			data = []
			if len(results):
				for i in range(len(results)):
					p = replace_rdf(results[i]["p"]["value"])
					if p not in data and find_year(results[i]["n"]["value"]):  # 且n有数字 年份
						data.append(p)
				return data
			else:
				return None
		except Exception as e:
			print("发生错误:", e)
			time.sleep(5)
			continue


# 查询一跳的时间约束并降序排列
def query_order_desc_1hop(entityID, one_hop_path):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p  ?n where{ ns:%s  %s ?x  ?p ?n.FILTER( CONTAINS(STR(?p), \".to\") || CONTAINS(STR(?p), \"date\") || CONTAINS(STR(?p), \".from\"))}" % (
		entityID, one_hop_path)
	print("------------------", sparql)

	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			data = []
			if len(results):
				for i in range(len(results)):
					p = replace_rdf(results[i]["p"]["value"])
					if p not in data and find_year(results[i]["n"]["value"]):  # 且n有数字 年份
						if "from" in p.split(".")[2] or "date" in p.split(".")[2] or "to" in p.split(".")[2]:
							data.append(p)
				print(data)
				return data
			else:
				return None
		except Exception as e:
			print("发生错误:", e)
			time.sleep(5)
			continue


# 查询第二跳的时间约束并升序排列
def query_order_asc_2hop(entityID, path):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p  ?n where{ ns:%s %s ?y ?p ?n.FILTER( CONTAINS(STR(?p), \".from\") || CONTAINS(STR(?p), \"date\") )}" % (
		entityID, path)
	# print(sparql)

	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			data = []
			if len(results):
				for i in range(len(results)):
					p = replace_rdf(results[i]["p"]["value"])
					if p not in data and find_year(results[i]["n"]["value"]):  # 且n有数字 年份
						if "from" in p.split(".")[2] or "date" in p.split(".")[2]:
							data.append(p)
				return data
			else:
				return None
		except Exception as e:
			print("发生错误:", e)
			time.sleep(5)
			continue


# 查询第二跳的时间约束并降序排列
def query_order_desc_2hop(entityID, path):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p  ?n where{ ns:%s %s ?y ?p ?n.FILTER( CONTAINS(STR(?p), \".to\") || CONTAINS(STR(?p), \"date\") || CONTAINS(STR(?p), \".from\") )}" % (
		entityID, path)
	# print(sparql)

	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			data = []
			if len(results):
				for i in range(len(results)):
					p = replace_rdf(results[i]["p"]["value"])
					if p not in data and find_year(results[i]["n"]["value"]):  # 且n有数字 年份
						if "from" in p.split(".")[2] or "date" in p.split(".")[2] or "to" in p.split(".")[2]:
							data.append(p)
				print(data)
				return data
			else:
				return None
		except Exception as e:
			print("发生错误:", e)
			time.sleep(5)
			continue


# 两跳时间约束
def query_2hop_p_from_to(entityID, two_path, yearData):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>  SELECT DISTINCT ?p ?o  WHERE { ns:%s %s ?y ?p ?o. FILTER(CONTAINS(STRAFTER(STRAFTER(STRAFTER(STRAFTER(STR(?p), \".\"), \".\"), \".\"), \".\"), \".from\") || CONTAINS(STRAFTER(STRAFTER(STRAFTER(STRAFTER(STR(?p), \".\"), \".\"), \".\"), \".\"), \".to\") || CONTAINS(STRAFTER(STRAFTER(STRAFTER(STRAFTER(STR(?p), \".\"), \".\"), \".\"), \".\"), \"date\"))}" % (
		entityID, two_path)  # 仅仅查找出包含 from 或者t o的路径
	print(sparql)

	while True:
		try:
			sparqlQuery.setQuery(sparql)
			results = sparqlQuery.query().convert()["results"]["bindings"]
			# print(results)
			from_path = ""
			to_path = ""
			if len(results):
				for i in range(len(results)):
					# 是否考虑多个from to的路径？
					p = replace_rdf(results[i]["p"]["value"])
					o = replace_rdf(results[i]["o"]["value"])
					if str(yearData) in o and ".from" in p:
						from_path = p
					if str(yearData) in o and ".to" in p:
						to_path = p
					if str(yearData) in o and "date" in p:
						# 针对"from_date"的路径
						if "from" in p:
							from_path = p
						elif "to" in p:
							to_path = p
						else:
							from_path = p
							to_path = p
				if from_path and to_path:
					if to_path.split(".")[1] == from_path.split(".")[1] and to_path.split(".")[0] == \
							from_path.split(".")[0]:
						return from_path, to_path
				return None, None
			else:
				return None, None
		except Exception as e:
			print("发生错误:", e)
			return None, None
		# time.sleep(10)
		# continue


# 将查出的from 和 to 配对 对于两跳均是加在y上的
def get_1hop_p_o_name(entityID):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p ?o ?n where{ ns:%s  ?p ?o.?o ns:type.object.name ?n}" % (
			entityID)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						po = [replace_rdf(results[i]["p"]["value"]), replace_rdf(results[i]["o"]["value"]),
							  replace_rdf(results[i]["n"]["value"])]
						if po not in data:
							data.append(po)
					# 返回不重复的Path
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 知道id，path，找o
def get_1hop_o(entityID, p):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select DISTINCT ?o  where{ ns:%s  ns:%s ?o}" % (
			entityID, p)
		with eventlet.Timeout(60, False):
			while True:
				try:
					sparqlQuery.setQuery(query_txt)
					results = sparqlQuery.query().convert()["results"]["bindings"]
					data = []
					if len(results):
						for i in range(len(results)):
							temp = replace_rdf(results[i]["o"]["value"])
							if temp not in data:
								data.append(temp)
						# 返回不重复的Path
						return data
					else:
						return None
				except Exception as e:
					print("发生错误:", e)
					time.sleep(5)
					continue
	return None


# 根据id和path 获取第二跳的路径
def from_id_path_get_2hop_po(entityID, one_hop_path):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select DISTINCT ?q where{ FILTER (?x != ns:%s) ns:%s %s ?y.?y ?q ?x}" % (
			entityID, entityID, one_hop_path)
		# print(query_txt)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				# 选择分数较高和top-k个关系
				if len(results):
					for i in range(len(results)):
						temp = replace_rdf(results[i]["q"]["value"])
						if temp not in data:
							data.append(temp)
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue

def from_id_path_get_2hop_po_top_k(entityID, one_hop_path, k):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select DISTINCT ?q where{ FILTER (?x != ns:%s) ns:%s %s ?y.?y ?q ?x}" % (
			entityID, entityID, one_hop_path)
		# print(query_txt)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				# 选择分数较高和top-k个关系
				if len(results):
					for i in range(len(results)):
						temp = replace_rdf(results[i]["q"]["value"])
						if temp not in data:
							data.append(temp)
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 生成n-gram list
def create_ngram_list(input):
	input_list = input.lower().split(" ")
	ngram_list = []
	for num in range(len(input_list) + 1, -1, -1):
		for tmp in zip(*[input_list[i:] for i in range(num)]):
			tmp = " ".join(tmp)
			ngram_list.append(tmp)
	return ngram_list


def is_substring_byword(name, question):
	return set(re.split(r"[/ -]", name.lower())) & set(re.split(r"[/ -]", question.lower()))


def is_substring(n, q):
	n_list = n.lower().split("/")  # College/University
	for ns in n_list:
		if ns in q.lower():
			return True
	return False


# 一跳答案实体约束id和名字的获取
"""先用ns:common.topic.notable_types去获取实体约束，后面再用模型选择路径和实体约束"""


# 一跳加在x上的实体约束
def from_id_path_get_2hop_po_oName(question, entityID, one_hop_path):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?y ?n  where{ FILTER (?x != ns:%s) ns:%s %s ?x ns:common.topic.notable_types ?y.  ?y ns:type.object.name ?n}" % (
			entityID, entityID, one_hop_path)
		print(query_txt)
		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						name = results[i]["n"]["value"].replace("@en", "")
						# 查出的name可能包含多个，这里加入选择模型
						# cons_path = replace_rdf(results[i]["q"]["value"])
						cons_path = "common.topic.notable_types"
						id = replace_rdf(results[i]["y"]["value"])
						# 这里就先按单词匹配吧
						if ([cons_path, id, name] not in data and is_substring_byword(name, question)):
							data.append([cons_path, id, name])
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 加入了约束选择模型
def from_id_path_get_2hop_po_oName_new(question, entityID, one_hop_path):
	if "m." in entityID or "g." in entityID:
		x_path = one_hop_path.replace("ns:", "").replace(" ?x .", "")
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?y ?n  where{ FILTER (?y != ns:%s) ns:%s %s ?x ?q ?y. ?y ns:type.object.name ?n}" % (
			entityID, entityID, one_hop_path)
		print(query_txt)
		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				# 完全子串
				data1 = []
				# 不完全子串
				data2 = []
				if len(results):
					for i in range(len(results)):
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						# 查出的name可能包含多个，这里加入选择模型
						cons_path = replace_rdf(results[i]["q"]["value"])
						id = replace_rdf(results[i]["y"]["value"])
						# 先加入完全是子串的
						if is_substring(name, question) and [cons_path, id, name] not in data1:
							data1.append([cons_path, id, name])
						elif [cons_path, id, name] not in data2 and is_substring_byword(name, question):
							data2.append([cons_path, id, name])
					# print(data1)
					# print(data2)
					if data1:
						for i in data1:
							if "common.topic.notable_types" == i[0]:
								return i
						new_data1 = []
						for i in data1:
							new_data1.append([i[0], i[1], i[2], cons_cosin_sim(x_path, i[0])])
						sort_data = sorted(new_data1, key=lambda x: x[-1], reverse=True)
						return sort_data[0]
					elif data2:
						for i in data2:
							if "common.topic.notable_types" == i[0]:
								return i
						new_data2 = []
						for i in data2:
							new_data2.append([i[0], i[1], i[2], cons_cosin_sim(x_path, i[0])])
						sort_data = sorted(new_data2, key=lambda x: x[-1], reverse=True)
						return sort_data[0]
					return None
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 两跳问题加在x上的约束
def from_id_path_get_3hop_po_oName_x(question, entityID, path):
	if "m." in entityID or "g." in entityID:
		# ?q  ?x ?n分别代表第二跳实体（也许是答案）的下一条关系，实体ID，实体名称
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?z ?n  where{ ns:%s %s ?x ns:common.topic.notable_types ?z.  ?z ns:type.object.name ?n}" % (
			entityID, path)
		# print(query_txt)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						if is_substring_byword(name, question):
							cons_path = "common.topic.notable_types"
							id = replace_rdf(results[i]["z"]["value"])
							if [cons_path, id, name] not in data:
								data.append([cons_path, id, name])
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 两跳问题加在x上的约束

def from_id_path_get_3hop_po_oName_x_new(question, entityID, path):
	if "m." in entityID or "g." in entityID:
		# ?y后面
		y_path1 = path.split(" ?y .?y ")[-1].replace("ns:", "").replace(" ?x .", "")
		# ?y前面
		y_path2 = path.split(" ?y .?y ")[0].replace("ns:", "")
		merge_path = ".".join(set(y_path1.split(".") + y_path2.split(".")))
		# print(y_path1,y_path2,merge_path)
		# ?q  ?x ?n分别代表第二跳实体（也许是答案）的下一条关系，实体ID，实体名称
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?z ?n  where{ FILTER (?q != ns:%s) ns:%s %s ?x ?q ?z.  ?z ns:type.object.name ?n}" % (
		y_path2, entityID, path)
		# print(query_txt)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data1 = []
				data2 = []
				if len(results):
					for i in range(len(results)):
						cons_path = replace_rdf(results[i]["q"]["value"])
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						id = replace_rdf(results[i]["z"]["value"])
						# 先加入完全是子串的
						if is_substring(name, question) and [cons_path, id, name] not in data1:
							data1.append([cons_path, id, name])
						elif [cons_path, id, name] not in data2 and is_substring_byword(name, question):
							data2.append([cons_path, id, name])
					# print(data1)
					# print(data2)
					if data1:
						for i in data1:
							if "common.topic.notable_types" == i[0]:
								return i
						new_data1 = []
						for i in data1:
							# 以前一跳路径来比较
							# new_data1.append([i[0], i[1], i[2], cons_cosin_sim(y_path1, i[0])])
							# 以整个主路径来比较
							new_data1.append([i[0], i[1], i[2], cons_cosin_sim(merge_path, i[0])])
						sort_data = sorted(new_data1, key=lambda x: x[-1], reverse=True)
						# print(sort_data)
						return sort_data[0]
					elif data2:
						for i in data2:
							if "common.topic.notable_types" == i[0]:
								return i
						new_data2 = []
						for i in data2:
							new_data2.append([i[0], i[1], i[2], cons_cosin_sim(merge_path, i[0])])
						sort_data = sorted(new_data2, key=lambda x: x[-1], reverse=True)
						# print(sort_data)
						return sort_data[0]
					return None
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 查询候选实体约束路径
def get_candid_entitycons(question, entityID, path, is_twohop=False):
	if "m." in entityID or "g." in entityID:
		if is_twohop:
			query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?n  where{ ns:%s %s ?x ?q ?z.  ?z ns:type.object.name ?n}" % (
				entityID, path)
		else:
			query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?n  where{ FILTER (?y != ns:%s) ns:%s ns:%s ?x . ?x ?q ?y. ?y ns:type.object.name ?n}" % (
				entityID, entityID, path)
		print(query_txt)
		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						# 查出的name可能包含多个，这里加入选择模型
						cons_path = replace_rdf(results[i]["q"]["value"])
						if is_substring_byword(name, question) and cons_path not in data:
							data.append(cons_path)
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue

# 查询候选的约束路径
def get_candid_con_path(question, entityID, path, is_twohop=False, is_con_two=False):
	if "m." in entityID or "g." in entityID:
		if is_twohop:
			# ?y后面
			y_path1 = path.split(" ?y .?y ")[-1].replace("ns:", "").replace(" ?x .", "")
			# ?y前面
			y_path2 = path.split(" ?y .?y ")[0].replace("ns:", "")
			if is_con_two:
				query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?n  where{ FILTER (?q != ns:%s) ns:%s %s ?x ?q ?z.  ?z ns:type.object.name ?n}" % (
					y_path1, entityID, path)
			else:
				query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?n  where{ FILTER (?q != ns:%s) FILTER (?q != ns:%s) ns:%s %s ?y ?q ?z.  ?z ns:type.object.name ?n}" % (
					y_path1, y_path2, entityID, path)
		else:
			query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?q ?n  where{ FILTER (?y != ns:%s) ns:%s ns:%s ?x . ?x ?q ?y. ?y ns:type.object.name ?n}" % (
				entityID, entityID, path)
		print(query_txt)
		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						# 查出的name可能包含多个，这里加入选择模型
						cons_path = replace_rdf(results[i]["q"]["value"])
						if cons_path not in data:
							data.append(cons_path)
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue

# 两跳问题加在y上的实体约束
"""
第一步，先用约束名整体去匹配问题子串，若存在则返回
第二步，切分问题为每个单词再去匹配问题子串，选择以中间实体周边一跳路径（与第二跳主路径不一致）与主路径的第一跳相似度最高的作为中间实体约束路径，再通过该模型对于该路径进行选择（因为同名路径很多，需要指定id才行）"""


def from_id_path_get_3hop_po_oName_y(question, entityID, path):
	if "m." in entityID or "g." in entityID:
		# path: "ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
		# ?y后面
		y_path1 = path.split(" ?y .?y ")[-1].replace("ns:", "").replace(" ?x .", "")
		# ?y前面
		y_path2 = path.split(" ?y .?y ")[0].replace("ns:", "")
		merge_path = ".".join(set(y_path1.split(".") + y_path2.split(".")))

		# print(y_path2)
		# print(y_path1)
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?q ?p ?n  where{ FILTER (?p != ns:%s) FILTER (?q != ns:%s) ns:%s %s ?y ?q ?p. ?p ns:type.object.name ?n}" % (
			entityID, y_path1, entityID, path)
		# print(query_txt)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						# 去掉一些疑问词
						name_new = name.replace("what", "").replace("when", "").replace("where", "").replace("why", "").replace(
							"how", "").replace("who", "").replace("which", "")
						# 先整体匹配
						if is_substring(name_new, question):
							cons_path = replace_rdf(results[i]["q"]["value"])
							id = replace_rdf(results[i]["p"]["value"])
							if [cons_path, id, name] not in data:
								data.append([cons_path, id, name])
					if data:
						return data[0]
					else:
						for i in range(len(results)):
							name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
							# 去掉一些疑问词
							name_new = name.replace("what", "").replace("when", "").replace("where", "").replace("why",
																												 "").replace(
								"how", "").replace("who", "").replace("which", "")
							# 后分为单词匹配
							if is_substring_byword(name_new, question):
								cons_path = replace_rdf(results[i]["q"]["value"])
								id = replace_rdf(results[i]["p"]["value"])
								if [cons_path, id, name] not in data:
									data.append([cons_path, id, name])
						if data:
							new_data = []
							for i in data:
								new_data.append([i[0], i[1], i[2], cons_cosin_sim(merge_path, i[0])])
							sort_data = sorted(new_data, key=lambda x: x[-1], reverse=True)
							return sort_data[0]
						else:
							return None
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


def from_id_path_get_3hop_po_oName_y_new(question, entityID, path):
	if "m." in entityID or "g." in entityID:
		# path: "ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
		# ?y后面
		y_path1 = path.split(" ?y .?y ")[-1].replace("ns:", "").replace(" ?x .", "")
		# ?y前面
		y_path2 = path.split(" ?y .?y ")[0].replace("ns:", "")
		merge_path = ".".join(set(y_path1.split(".") + y_path2.split(".")))

		# print(y_path2)
		# print(y_path1)
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?q ?p ?n  where{ FILTER (?p != ns:%s) FILTER (?q != ns:%s) ns:%s %s ?y ?q ?p. ?p ns:type.object.name ?n}" % (
			entityID, y_path1, entityID, path)
		# print(query_txt)
		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data1 = []
				data2 = []
				if len(results):
					for i in range(len(results)):
						cons_path = replace_rdf(results[i]["q"]["value"])
						name = results[i]["n"]["value"].replace("@en", "").replace("\"", "")
						id = replace_rdf(results[i]["p"]["value"])
						# 先加入完全是子串的
						if is_substring(name, question) and [cons_path, id, name] not in data1:
							data1.append([cons_path, id, name])
						elif [cons_path, id, name] not in data2 and is_substring_byword(name, question):
							data2.append([cons_path, id, name])
					# print(data1)
					# print(data2)
					if data1:
						# 类型约束只在答案结点上
						# for i in data1:
						# 	if "common.topic.notable_types" == i[0]:
						# 		return i
						new_data1 = []
						for i in data1:
							# 以前一跳路径来比较
							# new_data1.append([i[0], i[1], i[2], cons_cosin_sim(y_path1, i[0])])
							# 以整个主路径来比较
							new_data1.append([i[0], i[1], i[2], cons_cosin_sim(merge_path, i[0])])
						sort_data = sorted(new_data1, key=lambda x: x[-1], reverse=True)
						# print(sort_data)
						return sort_data[0]
					elif data2:
						# for i in data2:
						# 	if "common.topic.notable_types" == i[0]:
						# 		return i
						new_data2 = []
						for i in data2:
							new_data2.append([i[0], i[1], i[2], cons_cosin_sim(merge_path, i[0])])
						sort_data = sorted(new_data2, key=lambda x: x[-1], reverse=True)
						# print(sort_data)
						return sort_data[0]
					return None
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


def from_id_to_topK_path(query_e, entityId, k, threshold):
	result = get_1hop_p(entityId)  # 已知id获取路径
	candidate_path = []
	if result:
		candidate_path = cosin_sim(query_e, result)
	# print(candidate_path) #[('location.country.languages_spoken', 0.8907366991043091), ('location.country.official_language', 0.7186728715896606)]
	path_score = []
	# print(min(k, len(candidate_path)))
	if candidate_path:
		for i in range(min(k, len(candidate_path))):
			# if i == 0:
			path_score.append(candidate_path[i])
	# elif candidate_path[i][1] > threshold:  # 阈值！！！
	#     path_score.append(candidate_path[i])
	return path_score


# 获取两条的top-k个路径
def from_id_to_topK_2path(query_e, entityId, one_hop_rel, k, threshold):
	result = from_id_path_get_2hop_po(entityId, one_hop_rel)  # 已知id获取路径
	candidate_path = []
	if result:
		candidate_path = cosin_sim(query_e, result)
	# print(candidate_path) #[('location.country.languages_spoken', 0.8907366991043091), ('location.country.official_language', 0.7186728715896606)]
	path_score = []
	if candidate_path:
		for i in range(min(k, len(candidate_path))):
			# if i == 0:
			path_score.append(candidate_path[i])
	# elif candidate_path[i][1] > threshold:  # 阈值！！！
	# path_score.append(candidate_path[i])
	return path_score


def getConsPosPath(InferentialChain, sparql):
	x_cons = []
	y_cons = []
	for line in sparql.split("\n"):
		spo = line.split(" ", 4)
		if spo[0].split(":")[0] == "?x" and spo[2].split(":")[0] == "ns":
			x_cons.append(spo[1].split(":")[1])
		if spo[0].split(":")[0] == "?y" and spo[2].split(":")[0] == "ns":
			y_cons.append(spo[1].split(":")[1])
		if "FILTER(NOT" in line:
			y_cons.append(line.split(" ")[3].split(":")[1])
	if x_cons == []:
		x_cons = ""
	if y_cons == []:
		y_cons = ""

	return InferentialChain[0] + str(y_cons) + " [unused1] " + InferentialChain[1] + str(x_cons)


def get_2hop_po(entityID):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select  ?p ?y ?q ?x where{ ns:%s ?p ?y.?y ?q ?x}" % (
			entityID)
		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				data = []
				if len(results):
					for i in range(len(results)):
						temp = [replace_rdf(results[i]["p"]["value"]), replace_rdf(results[i]["q"]["value"])]
						if temp not in data:
							data.append(temp)
					return data
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


def query_virtuoso_entityName(entityID):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select DISTINCT ?x  where{ ns:%s  ns:type.object.name ?x}" % (
			entityID)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				if len(results):
					return results[0]["x"]["value"].replace("@en", "").replace("\"", "")
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


"""查询实体类型   type.object.name就是查询实体的名字， common.topic.notable_types才是查询实体类型"""


def query_virtuoso_entityType(entityID):
	if "m." in entityID or "g." in entityID:
		query_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> select DISTINCT ?n where{ ns:%s  ns:common.topic.notable_types ?x.?x ns:type.object.name  ?n}" % (
			entityID)

		while True:
			try:
				sparqlQuery.setQuery(query_txt)
				results = sparqlQuery.query().convert()["results"]["bindings"]
				if len(results):
					return results[0]["n"]["value"].replace("@en", "").replace("\"", "")
				else:
					return None
			except Exception as e:
				print("发生错误:", e)
				time.sleep(5)
				continue


# 获取答案实体的id
def get_answer(entityID, queryGraph):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?x WHERE { FILTER (?x != ns:%s) ns:%s %s }" % (
		entityID, entityID, queryGraph)
	print(sparql)
	sparqlQuery.setQuery(sparql)
	results = sparqlQuery.query().convert()["results"]["bindings"]
	# print(results)
	answer = []
	if len(results):
		for i in range(len(results)):
			data = replace_rdf(results[i]["x"]["value"])
			if data not in answer:
				answer.append(data)
		# print(answer)
		return answer
	else:
		return None


# 对于排序和时间等约束查询语句不一样
def get_answer_new(entityID, queryGraph):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?x WHERE { FILTER (?x != ns:%s) ns:%s %s " % (
		entityID, entityID, queryGraph)
	print(sparql)
	sparqlQuery.setQuery(sparql)
	results = sparqlQuery.query().convert()["results"]["bindings"]
	# print(results)
	answer = []
	if len(results):
		for i in range(len(results)):
			data = replace_rdf(results[i]["x"]["value"])
			if data not in answer:
				answer.append(data)
		# print(answer)
		return answer
	else:
		return None


# 针对答案是实体名字而非ID（如compQA）
def get_answer_name(entityID, queryGraph):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?n WHERE { FILTER (?x != ns:%s) ns:%s %s }" % (
		entityID, entityID, queryGraph)
	print(sparql)
	sparqlQuery.setQuery(sparql)
	results = sparqlQuery.query().convert()["results"]["bindings"]
	# print(results)
	answer = []
	if len(results):
		for i in range(len(results)):
			data = replace_rdf(results[i]["n"]["value"])
			data = data.replace("@en", "").replace("\"", "").lower()
			if data not in answer:
				answer.append(data)
		print(answer)
		return answer
	else:
		return None


# 针对答案是实体名字而非ID（如compQA）
def get_answer_name_new(entityID, queryGraph):
	sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?n WHERE { FILTER (?x != ns:%s) ns:%s %s " % (
		entityID, entityID, queryGraph)
	print(sparql)
	sparqlQuery.setQuery(sparql)
	results = sparqlQuery.query().convert()["results"]["bindings"]
	# print(results)
	answer = []
	if len(results):
		for i in range(len(results)):
			data = replace_rdf(results[i]["n"]["value"])
			data = data.replace("@en", "").replace("\"", "").lower()
			if data not in answer:
				answer.append(data)
		print(answer)
		return answer
	else:
		return None


# 获取答案实体的名字
def get_answer_to_name(entityID, queryGraph):
	try:
		sparql = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?x WHERE { FILTER (?x != ns:%s) . ns:%s ns:%s ?x}" % (
			entityID, entityID, queryGraph)
		# print(sparql)
		sparqlQuery.setQuery(sparql)
		results = sparqlQuery.query().convert()["results"]["bindings"]
		# print(results)
		answer = []
		answer_name = []
		if len(results):
			for i in range(len(results)):
				data = results[i]["x"]["value"]
				if data not in answer:
					if "http://rdf.freebase.com/ns/" in data:
						sparql_name = "PREFIX ns: <http://rdf.freebase.com/ns/> select ?n where{ ns:%s  ns:type.object.name  ?n}" % (
							replace_rdf(data))
						sparqlQuery.setQuery(sparql_name)
						results_name = sparqlQuery.query().convert()["results"]["bindings"]
						if len(results_name):
							name = results_name[0]["n"]["value"].replace("@en", "").replace("\"", "").lower()
							if name not in answer_name:
								answer_name.append(name)
			# print(answer_name)
			return answer_name
		else:
			return None
	except Exception:
		print("发生异常")
		return None


if __name__ == '__main__':
	# print(from_id_path_get_3hop_po_oName_y("what year did president william henry harrison take office?", "m.0835q", "ns:government.politician.government_positions_held ?y .?y ns:government.government_position_held.from ?x ."))
	# print(get_answer_name("m.0d3k14", "ns:government.us_president.vice_president ?x .?x ns:type.object.name ?n ."))
	# print(from_id_path_get_2hop_po_oName_new("what book did benjamin franklin published", "m.019fz", "ns:book.author.works_written ?x ."))
	print(from_id_path_get_3hop_po_oName_x_new("what high school did harper lee go to?", "m.01bq7x",
											   "ns:people.person.education ?y .?y ns:education.education.institution ?x ."))
