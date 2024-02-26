#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/17 19:45
# @Author  : Xavier Byrant
# @FileName: F1_test.py
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/7
# @Author  : hehl
# @Software: PyCharm
# @File    : candidate_queryGraph.py
import csv


import json
from os.path import join

import torch
from tqdm import tqdm
from loguru import logger

from model import SimcseModel_T5_GRU_CNN_Attention
from query_virtuoso import *
from train import parse_args_train
from train_beam_search import parse_args
from utils import cosin_sim
from utils_beam_search import cosin_sim_beam_search
from question_classification import question_cosin_sim, question_hop_num
import warnings




"""需要注意的是sparql的查询语言是否正确！！！"""


def FindInList(entry, elist):
	for item in elist:
		if entry == item:
			return True
	return False


# 计算问题和查询路径的得分
def Cal_q_path_score(question, paths):
	path_score = {}
	for p in paths:
		path_score[p] = cosin_sim(question, p)
	# return dict(sorted(path_score.items(), key=lambda x: x[1], reverse=True))
	return path_score


def hit_n(goldAnswerList, predAnswerList, n):
	if len(goldAnswerList) == 0:
		if predAnswerList is None:
			return True
		else:
			return False
	if predAnswerList is None:
		return False
	glist = [x["AnswerArgument"] for x in goldAnswerList]
	result = set(glist) & set(predAnswerList[:n])
	if result:
		return True
	else:
		return False


def hit_n_complexqa(goldAnswerList, predAnswerList, n):
	if len(goldAnswerList) == 0:
		if predAnswerList is None:
			return True
		else:
			return False
	if predAnswerList is None:
		return False
	result = set(goldAnswerList) & set(predAnswerList[:n])
	if result:
		return True
	else:
		return False
# 这里的计算F1方法是用答案实体的id号来匹配的
def CalculatePRF1(goldAnswerList, predAnswerList):
	# print(predAnswerList)
	# print(goldAnswerList)
	if len(goldAnswerList) == 0:
		if predAnswerList is None:
			return [1.0, 1.0,
					1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
		else:
			return [0.0, 1.0,
					0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
	elif predAnswerList is None:
		return [1.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
	else:
		glist = [x["AnswerArgument"] for x in goldAnswerList]
		plist = predAnswerList

		tp = 1e-40  # numerical trick
		fp = 0.0
		fn = 0.0

		for gentry in glist:
			if FindInList(gentry, plist):
				tp += 1
			else:
				fn += 1
		for pentry in plist:
			if not FindInList(pentry, glist):
				fp += 1

		precision = tp / (tp + fp)
		recall = tp / (tp + fn)

		f1 = (2 * precision * recall) / (precision + recall)
		return [precision, recall, f1]


def CalculatePRF1_C3PM(goldAnswerList, predAnswerList):
	# print(predAnswerList)
	# print(goldAnswerList)
	if len(goldAnswerList) == 0:
		if predAnswerList is None:
			return [1.0, 1.0,
					1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
		else:
			return [0.0, 1.0,
					0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
	elif predAnswerList is None:
		return [1.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
	else:
		glist = [x for x in goldAnswerList]
		plist = predAnswerList

		tp = 1e-40  # numerical trick
		fp = 0.0
		fn = 0.0

		for gentry in glist:
			if FindInList(gentry, plist):
				tp += 1
			else:
				fn += 1
		for pentry in plist:
			if not FindInList(pentry, glist):
				fp += 1

		precision = tp / (tp + fp)
		recall = tp / (tp + fn)

		f1 = (2 * precision * recall) / (precision + recall)
		return [precision, recall, f1]


def CQ_CalculatePRF1(goldAnswerList, predAnswerList):
	# print(predAnswerList)
	if len(goldAnswerList) == 0:
		if predAnswerList is None:
			return [1.0, 1.0,
					1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
		else:
			return [0.0, 1.0,
					0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
	elif predAnswerList is None:
		return [1.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
	else:
		glist = goldAnswerList
		plist = predAnswerList

		tp = 1e-40  # numerical trick
		fp = 0.0
		fn = 0.0

		for gentry in glist:
			if FindInList(gentry, plist):
				tp += 1
			else:
				fn += 1
		for pentry in plist:
			if not FindInList(pentry, glist):
				fp += 1

		precision = tp / (tp + fp)
		recall = tp / (tp + fn)

		f1 = (2 * precision * recall) / (precision + recall)
		return [precision, recall, f1]



# 获取SPARQL语句中的查询图
def clean_sparql(topic_entity, sparql):
	queryGraph = ""
	oneSparql = sparql.strip().replace(f"ns:{topic_entity}", "").split("\n")
	for i in range(0, len(oneSparql)):
		if i >= 4:  # 去除前面部分的sparql  直接获取查询图
			# 删除主题实体
			if i == len(oneSparql) - 1 and oneSparql[i] == "}":  # 若最后一个是}则 不加入！
				pass
			else:
				queryGraph = queryGraph + oneSparql[i].split("#")[0]
	return queryGraph


"""针对查询图 选择能查询出答案的为正样本 若完全正确的则为1 否则为0.7"""

"""这里生成的查询图是按照步骤（主题实体-->主路径-->约束）来的，先试试效果，后期再考虑从文件获取打好分的主路径，只需要选择最高的主路径再加约束就去查库计算F1"""
"""或者还是按照步骤来，只需要调一个约束选择的包来选择约束"""
"""训练集有些是没有InferentialChain需要跳过"""

# n:引入每一类随机问题个数，k:取前k个看占比
# def webQSP_answer_predict(name, args, n, k):
# 	# top_id_name_tname = return_type_topid()
# 	# print(f"测试集中有实体id的长度为：{len(top_id_name_tname)}")
# 	type_num = 0
# 	preEntity_num = 0
#
# 	gender_male = ["dad", "father", "son", "brothers", "brother"]
# 	gender_female = ["mom", "daughter", "wife", "mother", "mum"]
# 	marry = ["husband", "married", "marry", "wife"]  # people.marriage.type_of_union
# 	order_ASC = "first"  # from start_date  ?y ns:sports.sports_team_roster.from ?sk0 . ORDER BY xsd:datetime(?sk0) LIMIT 1    #first name/ first wife /first language
# 	order_DESC = "last"  # last year ! = 2014
# 	# other_DESC = ["the most", "biggest", "largest", "predominant", "tallest", "major", "newly"]
#
# 	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
# 	with open(data_path, "r", encoding="utf-8") as f:
# 		data = json.load(f)
# 		# 没有答案的问题
# 		"""生成查询图"""
# 		# 存储问题id和答案
# 		preAnswer_list = []
# 		for i in tqdm(data["Questions"]):
# 			QuestionId = i["QuestionId"]
# 			# if int(q_id.replace("WebQTest-", "")) > -1 :
# 			# if int(q_id.replace("WebQTrn-", "")) > -1:
#
# 			# 存储问题的基本信息 id 问题 提及 实体id
# 			if i["Parses"][0]["PotentialTopicEntityMention"] and i["Parses"][0]["TopicEntityMid"] and i["Parses"][0][
# 				"TopicEntityName"] and i["Parses"][0]["Sparql"]:  # 一定数据要都存在再处理
# 				question = i["ProcessedQuestion"].lower()  # 大写转小写
# 				mention = i["Parses"][0]["PotentialTopicEntityMention"].lower()
# 				topicEntityID = i["Parses"][0]["TopicEntityMid"]
# 				topicEntityName = i["Parses"][0]["TopicEntityName"]
# 				Answersinfo = i["Parses"][0]["Answers"]
# 				# 这里记录的是答案的id号
# 				Answers = []
# 				if Answersinfo:
# 					for answer in Answersinfo:
# 						Answers.append(answer["AnswerArgument"])
# 				"存储问题的实体类型 由于前期已经查询过 直接从文件获取 不再查库浪费时间 "
# 				# 获得主题实体类型名
# 				# topicEntityType = top_id_name_tname[QuestionId]["topicEntityTypeName"]
# 				# Sparql = i["Parses"][0]["Sparql"]
# 				"PosQueryGraph应该是正确的查询图"
# 				if i["Parses"][0]["InferentialChain"]:
# 					PosQueryGraph = i["Parses"][0]["InferentialChain"]
# 				# else:
# 				# 	hop_num = question_hop_num(question, n, k)
# 				# hop_num = question_hop_num(question, n, k) + 1
# 				# print(f"问题{question}实际跳数为{len(PosQueryGraph)}，预测跳数为{hop_num}")
# 				preAnswer = {}
# 				preAnswer["QuestionId"] = QuestionId
# 				query_sent = ""
# 				query_e = question.replace(mention, "<e>")
# 				# 返回的是问题的year list
# 				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
# 				one_hop_rels = get_1hop_p(topicEntityID)
# 				# 无约束
# 				if one_hop_rels:
# 					one_hop_nocons_data = []
# 					is_use_entity_cons = True
# 					"""查找一跳无约束"""
# 					for one_hop_rel in one_hop_rels:
# 						cos = cosin_sim(query_e, one_hop_rel, args)
# 						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
# 						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
# 					if one_hop_nocons_data:
# 						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
# 					else:
# 						preAnswer["Answers"] = []
# 						preAnswer_list.append(preAnswer)
# 						continue
# 					# 这里暂且对top k个查询图进行后续约束加入
# 					if len(PosQueryGraph) == 1:
# 					# if hop_num == 1:
# 						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
# 						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
# 						query_sent = one_hop[1]
# 						"男性约束"
# 						if set(gender_male) & set(query_e.split()):
# 							# one_hop[1]格式"ns:" + one_hop_rel + " ?x ."
# 							query_sent = one_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
# 							is_use_entity_cons = False
#
# 						"女性约束"
# 						if set(gender_female) & set(query_e.split()):
# 							query_sent = one_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
# 							is_use_entity_cons = False
# 						"""实体约束"""
# 						if is_use_entity_cons:
# 							# 这里一跳的实体约束还可以再改进一下按照中间实体约束那样搞（后面再弄）
# 							one_hop_entity_constrains = from_id_path_get_2hop_po_oName(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
# 							if one_hop_entity_constrains:
# 								# 这里可以对候选实体约束排个序选最高
# 								one_hop_entity_constrain = one_hop_entity_constrains[0]
# 								# one_hop_entity_constrain = one_hop_entity_constrains
# 								# [core_path,id.name] 这里是加完其他约束后根据情况再加实体约束
# 								query_sent = query_sent + f"?x ns:{one_hop_entity_constrain[0]} ns:{one_hop_entity_constrain[1]} ."
# 						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
# 						if yearCons:  # 如果问题中有年份的时间约束
# 							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
# 							# 至于约束的名称则需要在库里面查询 （ from ,to ）
# 							from_path, to_path = query_1hop_p_from_to(topicEntityID, one_hop[1], yearData)  # 选出成对的路径
# 							if from_path and to_path:
# 								query_sent = query_sent + 'FILTER(NOT EXISTS {?x ns:%s ?sk0} || EXISTS {?x ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
# 														  'FILTER(NOT EXISTS {?x ns:%s ?sk2} || EXISTS {?x ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
# 											 % (from_path, from_path, yearData, to_path, to_path, yearData)
# 						# 升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）
# 						if order_ASC in query_e:
# 							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
# 							paths = query_order_asc_1hop(topicEntityID, one_hop[1])
# 							if paths:
# 								# 暂时只要一个
# 								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
# 									paths[0])
# 						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
# 						elif order_DESC in query_e:
# 							paths = query_order_desc_1hop(topicEntityID, one_hop[1])
# 							# print(paths) #多个path
# 							if paths:
# 								if "end_date" in paths:
# 									desc_path = "end_date"
# 								elif "start_date" in paths:
# 									desc_path = "start_date"
# 								else:
# 									desc_path = paths[0]
# 								# 暂时只要一个
# 								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
# 									desc_path)
# 						if query_sent[-1] == '1' or query_sent[-1] == '}':
# 							preAnswer["Answers"] = get_answer_new(topicEntityID, query_sent)
# 						else:
# 							preAnswer["Answers"] = get_answer(topicEntityID, query_sent)
# 						preAnswer_list.append(preAnswer)
# 					elif len(PosQueryGraph) == 2:
# 					# elif hop_num == 2:
# 						one_hop_in_two = one_hop_rels_sortbycos[0][1].replace(" ?x .", "")
# 						two_hop_rels = from_id_path_get_2hop_po(topicEntityID,
# 																one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
# 						if two_hop_rels:
# 							two_hop_nocons_data = []
# 							is_use_entity_cons = True
# 							for two_hop_rel in two_hop_rels:
# 								"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
# 								two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
# 								# 注意这里计算的是第二跳路径，没有两跳一起计算
# 								cos = cosin_sim(query_e, two_hop_rel, args_train)
# 								# 两条路径合并一起计算
# 								# cos = cosin_sim(query_e, two_hop)
# 								two_hop_nocons_data.append([query_e, two_hop, cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
# 							if two_hop_nocons_data:
# 								two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2], reverse=True)
# 							else:
# 								preAnswer["Answers"] = []
# 								preAnswer_list.append(preAnswer)
# 								continue
# 							"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
# 							"""同样最后处理实体约束, 同样选前k个"""
# 							# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
# 							two_hop = two_hop_rels_sortbycos[0]  # [query_e, two_hop_rel, cos, F1]
# 							query_sent = two_hop[1]
#
# 							"""男性约束"""
# 							if set(gender_male) & set(query_e.split()):
# 								query_sent = two_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
# 								is_use_entity_cons = False
#
# 							"""女性约束"""
# 							if set(gender_female) & set(query_e.split()):
# 								query_sent = two_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
# 								is_use_entity_cons = False
# 							"是否结婚的约束"
# 							if set(marry) & set(query_e.split()):
# 								query_sent = two_hop[
# 												 1] + f"?y ns:people.marriage.type_of_union ns:m.04ztj ."  # 婚姻 约束在y上 是否都有时间限制
# 							"""实体约束"""
# 							if is_use_entity_cons:
# 								"""处理约束 实体约束 约束加载第二跳的实体上"""  # 实体存在问题中 查询出第二跳的实体名称及路径
# 								two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x(question,
# 																							   topicEntityID,
# 																							   two_hop[1])
# 								"两跳实体约束（加在x上的）加了约束选择"
# 								# if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
# 								# 	two_hop_entity_constrain = two_hop_entity_constrains_x
# 								# 	# two_hop_entity_constrain:[path, id, name, score]
# 								# 	query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
#
# 								# 多个约束
# 								if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
# 									for two_hop_entity_constrain in two_hop_entity_constrains_x:
# 										# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
# 										query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
# 								# 在y上的实体约束
# 								"两跳实体约束（加在y上的）"
# 								two_hop_entity_constrains_y = from_id_path_get_3hop_po_oName_y(question,
# 																							   topicEntityID,
# 																							   two_hop[1])
# 								if two_hop_entity_constrains_y:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
# 									two_hop_entity_constrain = two_hop_entity_constrains_y
# 									# two_hop_entity_constrain:[path, id, name]
# 									query_sent = query_sent + f"?y ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
#
# 							"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
# 							if yearCons:  # 如果问题中有年份的时间约束
# 								yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
# 								# 至于约束的名称则需要在库里面查询 （ from ,to ）
# 								from_path, to_path = query_2hop_p_from_to(topicEntityID, two_hop[1],
# 																		  yearData)  # 选出成对的路径
# 								if from_path and to_path:
# 									# 数据集中gender之后没有year约束但是有DESC约束
# 									query_sent = query_sent + 'FILTER(NOT EXISTS {?y ns:%s ?sk0} || EXISTS {?y ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
# 															  'FILTER(NOT EXISTS {?y ns:%s ?sk2} || EXISTS {?y ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
# 												 % (from_path, from_path, yearData, to_path, to_path, yearData)
#
# 							"""升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）"""
# 							if order_ASC in query_e:
# 								# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
# 								paths = query_order_asc_2hop(topicEntityID, two_hop[1])
# 								if paths:
# 									# 暂时只要一个
# 									query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
# 										paths[0])
#
# 							# if F1 != 0:
# 							# 	is_use_entity_cons = False
#
# 							# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
# 							elif order_DESC in query_e:
# 								paths = query_order_desc_2hop(topicEntityID, two_hop[1])
# 								# print(paths) #多个path
# 								if paths:
# 									desc_path = paths[0]
# 									for path in paths:
# 										if "end_date" in paths:
# 											desc_path = path
# 											break
# 									# 暂时只要一个
# 									query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
# 										desc_path)
# 							print(query_sent)
# 							if query_sent[-1] == '1' or query_sent[-1] == '}':
# 								preAnswer["Answers"] = get_answer_new(topicEntityID, query_sent)
# 							else:
# 								preAnswer["Answers"] = get_answer(topicEntityID, query_sent)
# 							preAnswer_list.append(preAnswer)
# 					else:
# 						preAnswer["Answers"] = []
# 						preAnswer_list.append(preAnswer)
# 		with open(f"{args.output_path}/WebQSP_{name}_qid_preAnswer.json", 'w', encoding="utf-8") as f:
# 			json.dump(preAnswer_list, f)
# 			print("预测答案已写入文件，正在计算F1...")

def webQSP_answer_predict_beam_search_topk(name, args_train, args, k):
	# top_id_name_tname = return_type_topid()
	# print(f"测试集中有实体id的长度为：{len(top_id_name_tname)}")
	type_num = 0
	preEntity_num = 0
	print(f"正在评估超参数beam size:{k}, kesai:{args.kesai}的性能")
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	pt_path_train = join(args_train.output_path, "simcse.pt")
	model_train = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args_train.pretrain_model_path, pooling="cls", kesai=args_train.kesai, dropout=args_train.dropout).to(args_train.device)
	model_train.load_state_dict(torch.load(pt_path_train))

	gender_male = ["dad", "father", "son", "brothers", "brother"]
	gender_female = ["mom", "daughter", "wife", "mother", "mum"]
	marry = ["husband", "married", "marry", "wife"]  # people.marriage.type_of_union
	order_ASC = "first"  # from start_date  ?y ns:sports.sports_team_roster.from ?sk0 . ORDER BY xsd:datetime(?sk0) LIMIT 1    #first name/ first wife /first language
	order_DESC = "last"  # last year ! = 2014
	# other_DESC = ["the most", "biggest", "largest", "predominant", "tallest", "major", "newly"]

	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		preAnswer_list = []
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
				Answersinfo = i["Parses"][0]["Answers"]
				# 这里记录的是答案的id号
				Answers = []
				if Answersinfo:
					for answer in Answersinfo:
						Answers.append(answer["AnswerArgument"])
				"存储问题的实体类型 由于前期已经查询过 直接从文件获取 不再查库浪费时间 "
				# 获得主题实体类型名
				# topicEntityType = top_id_name_tname[QuestionId]["topicEntityTypeName"]
				# Sparql = i["Parses"][0]["Sparql"]
				"PosQueryGraph应该是正确的查询图"
				if i["Parses"][0]["InferentialChain"]:
					PosQueryGraph = i["Parses"][0]["InferentialChain"]

				preAnswer = {}
				preAnswer["QuestionId"] = QuestionId
				query_e = question.replace(mention, "<e>")
				# 返回的是问题的year list
				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
				one_hop_rels = get_1hop_p(topicEntityID)
				# 无约束
				if one_hop_rels:
					one_hop_nocons_data = []
					one_hop_rels_sortbycos = []
					is_use_entity_cons = True
					"""查找一跳无约束"""
					for one_hop_rel in one_hop_rels:
						cos = cosin_sim(query_e, one_hop_rel, args_train, model_train)
						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
					if one_hop_nocons_data:
						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
					else:
						preAnswer["Answers"] = []
						preAnswer_list.append(preAnswer)
						continue
					# 这里暂且对top k个查询图进行后续约束加入
					if len(PosQueryGraph) == 1:
						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
						query_sent = one_hop[1]
						"男性约束"
						if set(gender_male) & set(query_e.split()):
							# one_hop[1]格式"ns:" + one_hop_rel + " ?x ."
							query_sent = one_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							is_use_entity_cons = False

						"女性约束"
						if set(gender_female) & set(query_e.split()):
							query_sent = one_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							is_use_entity_cons = False
						"""实体约束"""
						if is_use_entity_cons:
							# 这里一跳的实体约束还可以再改进一下按照中间实体约束那样搞（后面再弄）
							one_hop_entity_constrains = from_id_path_get_2hop_po_oName(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
							if one_hop_entity_constrains:
								# 这里可以对候选实体约束排个序选最高
								one_hop_entity_constrain = one_hop_entity_constrains[0]
								# one_hop_entity_constrain = one_hop_entity_constrains
								# [core_path,id.name] 这里是加完其他约束后根据情况再加实体约束
								query_sent = query_sent + f"?x ns:{one_hop_entity_constrain[0]} ns:{one_hop_entity_constrain[1]} ."
						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_1hop_p_from_to(topicEntityID, one_hop[1], yearData)  # 选出成对的路径
							if from_path and to_path:
								query_sent = query_sent + 'FILTER(NOT EXISTS {?x ns:%s ?sk0} || EXISTS {?x ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														  'FILTER(NOT EXISTS {?x ns:%s ?sk2} || EXISTS {?x ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											 % (from_path, from_path, yearData, to_path, to_path, yearData)
						# 升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_1hop(topicEntityID, one_hop[1])
							if paths:
								# 暂时只要一个
								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
									paths[0])
						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
						elif order_DESC in query_e:
							paths = query_order_desc_1hop(topicEntityID, one_hop[1])
							# print(paths) #多个path
							if paths:
								if "end_date" in paths:
									desc_path = "end_date"
								elif "start_date" in paths:
									desc_path = "start_date"
								else:
									desc_path = paths[0]
								# 暂时只要一个
								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
									desc_path)
						if query_sent[-1] == '1' or query_sent[-1] == '}':
							preAnswer["Answers"] = get_answer_new(topicEntityID, query_sent)
						else:
							preAnswer["Answers"] = get_answer(topicEntityID, query_sent)
						preAnswer_list.append(preAnswer)
					elif len(PosQueryGraph) == 2:
						one_hop_rels_top_k = one_hop_rels_sortbycos[:k]
						# beam search找到的top k个路径，还未整体排序
						two_hop_all_candid = []
						# 两跳打了分之后
						two_hop_all_scored = []
						# 两跳打了分并排序
						for one_hop_in_two in one_hop_rels_top_k:
							one_hop_in_two = one_hop_in_two[1].replace(" ?x .", "")
							# 当前one_hop_in_two
							two_hop_rels = from_id_path_get_2hop_po(topicEntityID, one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
							if two_hop_rels:
								two_hop_nocons_data = []
								is_use_entity_cons = True
								for two_hop_rel in two_hop_rels:
									"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
									two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
									# 注意这里计算的是第二跳路径，没有两跳一起计算
									cos = cosin_sim(query_e, two_hop_rel, args_train, model_train)
									# 两条路径合并一起计算
									# cos = cosin_sim(query_e, two_hop)
									# 问题和当前one_hop_in_two第二跳路径的cos得分
									two_hop_nocons_data.append([query_e, two_hop, cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
								if two_hop_nocons_data:
									two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2], reverse=True)
									two_hop_all_candid += two_hop_rels_sortbycos[:k]
							else:
								continue
						if not two_hop_all_candid:
							preAnswer["Answers"] = []
							preAnswer_list.append(preAnswer)
							continue
						else:
							# print(two_hop_all_candid)
							for two_hop in two_hop_all_candid:
								cos = cosin_sim_beam_search(query_e, two_hop[1], args, model_beam)
								two_hop_all_scored.append([two_hop[0], two_hop[1], cos])
							two_hop_all_scored_sortedbycos = sorted(two_hop_all_scored, key=lambda x: x[2], reverse=True)
						"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
						"""同样最后处理实体约束, 同样选前k个"""
						# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						two_hop = two_hop_all_scored_sortedbycos[0]  # [query_e, two_hop_rel, cos]
						query_sent = two_hop[1]

						"""男性约束"""
						if set(gender_male) & set(query_e.split()):
							query_sent = two_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							is_use_entity_cons = False

						"""女性约束"""
						if set(gender_female) & set(query_e.split()):
							query_sent = two_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							is_use_entity_cons = False
						"是否结婚的约束"
						if set(marry) & set(query_e.split()):
							query_sent = two_hop[
											 1] + f"?y ns:people.marriage.type_of_union ns:m.04ztj ."  # 婚姻 约束在y上 是否都有时间限制
						"""实体约束"""
						if is_use_entity_cons:
							"""处理约束 实体约束 约束加载第二跳的实体上"""  # 实体存在问题中 查询出第二跳的实体名称及路径
							two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x(question,
																						   topicEntityID,
																						   two_hop[1])
							"两跳实体约束（加在x上的）加了约束选择"
							# if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
							# 	two_hop_entity_constrain = two_hop_entity_constrains_x
							# 	# two_hop_entity_constrain:[path, id, name, score]
							# 	query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."

							# 多个约束
							if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
								for two_hop_entity_constrain in two_hop_entity_constrains_x:
									# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
									query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
							# 在y上的实体约束
							"两跳实体约束（加在y上的）"
							two_hop_entity_constrains_y = from_id_path_get_3hop_po_oName_y(question,
																						   topicEntityID,
																						   two_hop[1])
							if two_hop_entity_constrains_y:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
								two_hop_entity_constrain = two_hop_entity_constrains_y
								# two_hop_entity_constrain:[path, id, name]
								query_sent = query_sent + f"?y ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."

						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_2hop_p_from_to(topicEntityID, two_hop[1],
																	  yearData)  # 选出成对的路径
							if from_path and to_path:
								# 数据集中gender之后没有year约束但是有DESC约束
								query_sent = query_sent + 'FILTER(NOT EXISTS {?y ns:%s ?sk0} || EXISTS {?y ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														  'FILTER(NOT EXISTS {?y ns:%s ?sk2} || EXISTS {?y ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											 % (from_path, from_path, yearData, to_path, to_path, yearData)

						"""升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）"""
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_2hop(topicEntityID, two_hop[1])
							if paths:
								# 暂时只要一个
								query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
									paths[0])

						# if F1 != 0:
						# 	is_use_entity_cons = False

						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
						elif order_DESC in query_e:
							paths = query_order_desc_2hop(topicEntityID, two_hop[1])
							# print(paths) #多个path
							if paths:
								desc_path = paths[0]
								for path in paths:
									if "end_date" in paths:
										desc_path = path
										break
								# 暂时只要一个
								query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
									desc_path)
						print(query_sent)
						if query_sent[-1] == '1' or query_sent[-1] == '}':
							preAnswer["Answers"] = get_answer_new(topicEntityID, query_sent)
						else:
							preAnswer["Answers"] = get_answer(topicEntityID, query_sent)
						preAnswer_list.append(preAnswer)
				else:
					preAnswer["Answers"] = []
					preAnswer_list.append(preAnswer)
	with open(f"{args.output_path}/WebQSP_{name}_qid_preAnswer_k_{k}_kesai_{args.kesai}.json", 'w', encoding="utf-8") as f:
		json.dump(preAnswer_list, f)
		print("预测答案已写入文件，正在计算F1...")

def webQSP_answer_predict_beam_search_topk_only_constrained(name, args_train, args, k):
	# top_id_name_tname = return_type_topid()
	# print(f"测试集中有实体id的长度为：{len(top_id_name_tname)}")
	type_num = 0
	preEntity_num = 0
	print(f"正在评估超参数beam size:{k}, kesai:{args.kesai}的性能")
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	pt_path_train = join(args_train.output_path, "simcse.pt")
	model_train = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args_train.pretrain_model_path, pooling="cls", kesai=args_train.kesai, dropout=args_train.dropout).to(args_train.device)
	model_train.load_state_dict(torch.load(pt_path_train))

	gender_male = ["dad", "father", "son", "brothers", "brother"]
	gender_female = ["mom", "daughter", "wife", "mother", "mum"]
	marry = ["husband", "married", "marry", "wife"]  # people.marriage.type_of_union
	order_ASC = "first"  # from start_date  ?y ns:sports.sports_team_roster.from ?sk0 . ORDER BY xsd:datetime(?sk0) LIMIT 1    #first name/ first wife /first language
	order_DESC = "last"  # last year ! = 2014
	# other_DESC = ["the most", "biggest", "largest", "predominant", "tallest", "major", "newly"]

	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		preAnswer_list = []
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
				Answersinfo = i["Parses"][0]["Answers"]
				# 这里记录的是答案的id号
				Answers = []
				if Answersinfo:
					for answer in Answersinfo:
						Answers.append(answer["AnswerArgument"])
				"存储问题的实体类型 由于前期已经查询过 直接从文件获取 不再查库浪费时间 "
				# 获得主题实体类型名
				# topicEntityType = top_id_name_tname[QuestionId]["topicEntityTypeName"]
				# Sparql = i["Parses"][0]["Sparql"]
				"PosQueryGraph应该是正确的查询图"
				if i["Parses"][0]["InferentialChain"]:
					PosQueryGraph = i["Parses"][0]["InferentialChain"]

				preAnswer = {}
				preAnswer["QuestionId"] = QuestionId
				query_e = question.replace(mention, "<e>")
				# 返回的是问题的year list
				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
				one_hop_rels = get_1hop_p(topicEntityID)
				# 无约束
				if one_hop_rels:
					one_hop_nocons_data = []
					one_hop_rels_sortbycos = []
					is_use_entity_cons = True
					"""查找一跳无约束"""
					for one_hop_rel in one_hop_rels:
						cos = cosin_sim(query_e, one_hop_rel, args_train, model_train)
						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
					if one_hop_nocons_data:
						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
					else:
						preAnswer["Answers"] = []
						preAnswer_list.append(preAnswer)
						continue
					# 这里暂且对top k个查询图进行后续约束加入
					if len(PosQueryGraph) == 1:
						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
						query_sent = one_hop[1]
						"男性约束"
						if set(gender_male) & set(query_e.split()):
							# one_hop[1]格式"ns:" + one_hop_rel + " ?x ."
							query_sent = one_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							is_use_entity_cons = False

						"女性约束"
						if set(gender_female) & set(query_e.split()):
							query_sent = one_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							is_use_entity_cons = False
						"""实体约束"""
						if is_use_entity_cons:
							# 这里一跳的实体约束还可以再改进一下按照中间实体约束那样搞（后面再弄）
							one_hop_entity_constrains = from_id_path_get_2hop_po_oName(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
							if one_hop_entity_constrains:
								# 这里可以对候选实体约束排个序选最高
								one_hop_entity_constrain = one_hop_entity_constrains[0]
								# one_hop_entity_constrain = one_hop_entity_constrains
								# [core_path,id.name] 这里是加完其他约束后根据情况再加实体约束
								query_sent = query_sent + f"?x ns:{one_hop_entity_constrain[0]} ns:{one_hop_entity_constrain[1]} ."
						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_1hop_p_from_to(topicEntityID, one_hop[1], yearData)  # 选出成对的路径
							if from_path and to_path:
								query_sent = query_sent + 'FILTER(NOT EXISTS {?x ns:%s ?sk0} || EXISTS {?x ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														  'FILTER(NOT EXISTS {?x ns:%s ?sk2} || EXISTS {?x ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											 % (from_path, from_path, yearData, to_path, to_path, yearData)
						# 升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_1hop(topicEntityID, one_hop[1])
							if paths:
								# 暂时只要一个
								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
									paths[0])
						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
						elif order_DESC in query_e:
							paths = query_order_desc_1hop(topicEntityID, one_hop[1])
							# print(paths) #多个path
							if paths:
								if "end_date" in paths:
									desc_path = "end_date"
								elif "start_date" in paths:
									desc_path = "start_date"
								else:
									desc_path = paths[0]
								# 暂时只要一个
								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
									desc_path)
						if query_sent[-1] == '1' or query_sent[-1] == '}':
							preAnswer["Answers"] = get_answer_new(topicEntityID, query_sent)
						else:
							preAnswer["Answers"] = get_answer(topicEntityID, query_sent)
						preAnswer_list.append(preAnswer)
					elif len(PosQueryGraph) == 2:
						one_hop_rels_top_k = one_hop_rels_sortbycos[:k]
						# beam search找到的top k个路径，还未整体排序
						two_hop_all_candid = []
						# 两跳打了分之后
						two_hop_all_scored = []
						# 两跳打了分并排序
						for one_hop_in_two in one_hop_rels_top_k:
							one_hop_in_two = one_hop_in_two[1].replace(" ?x .", "")
							# 当前one_hop_in_two
							two_hop_rels = from_id_path_get_2hop_po(topicEntityID, one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
							if two_hop_rels:
								two_hop_nocons_data = []
								is_use_entity_cons = True
								for two_hop_rel in two_hop_rels:
									"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
									two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
									# 注意这里计算的是第二跳路径，没有两跳一起计算
									cos = cosin_sim(query_e, two_hop_rel, args_train, model_train)
									# 两条路径合并一起计算
									# cos = cosin_sim(query_e, two_hop)
									# 问题和当前one_hop_in_two第二跳路径的cos得分
									two_hop_nocons_data.append([query_e, two_hop, cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
								if two_hop_nocons_data:
									two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2], reverse=True)
									two_hop_all_candid += two_hop_rels_sortbycos[:k]
							else:
								continue
						if not two_hop_all_candid:
							preAnswer["Answers"] = []
							preAnswer_list.append(preAnswer)
							continue
						else:
							# print(two_hop_all_candid)
							for two_hop in two_hop_all_candid:
								cos = cosin_sim_beam_search(query_e, two_hop[1], args, model_beam)
								two_hop_all_scored.append([two_hop[0], two_hop[1], cos])
							two_hop_all_scored_sortedbycos = sorted(two_hop_all_scored, key=lambda x: x[2], reverse=True)
						"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
						"""同样最后处理实体约束, 同样选前k个"""
						# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						two_hop = two_hop_all_scored_sortedbycos[0]  # [query_e, two_hop_rel, cos]
						query_sent = two_hop[1]

						"""男性约束"""
						if set(gender_male) & set(query_e.split()):
							query_sent = two_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							is_use_entity_cons = False

						"""女性约束"""
						if set(gender_female) & set(query_e.split()):
							query_sent = two_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							is_use_entity_cons = False
						"是否结婚的约束"
						if set(marry) & set(query_e.split()):
							query_sent = two_hop[
											 1] + f"?y ns:people.marriage.type_of_union ns:m.04ztj ."  # 婚姻 约束在y上 是否都有时间限制
						"""实体约束"""
						if is_use_entity_cons:
							"""处理约束 实体约束 约束加载第二跳的实体上"""  # 实体存在问题中 查询出第二跳的实体名称及路径
							two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x(question,
																						   topicEntityID,
																						   two_hop[1])
							"两跳实体约束（加在x上的）加了约束选择"
							# if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
							# 	two_hop_entity_constrain = two_hop_entity_constrains_x
							# 	# two_hop_entity_constrain:[path, id, name, score]
							# 	query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."

							# 多个约束
							if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
								for two_hop_entity_constrain in two_hop_entity_constrains_x:
									# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
									query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
							# 在y上的实体约束
							"两跳实体约束（加在y上的）"
							two_hop_entity_constrains_y = from_id_path_get_3hop_po_oName_y(question,
																						   topicEntityID,
																						   two_hop[1])
							if two_hop_entity_constrains_y:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
								two_hop_entity_constrain = two_hop_entity_constrains_y
								# two_hop_entity_constrain:[path, id, name]
								query_sent = query_sent + f"?y ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."

						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_2hop_p_from_to(topicEntityID, two_hop[1],
																	  yearData)  # 选出成对的路径
							if from_path and to_path:
								# 数据集中gender之后没有year约束但是有DESC约束
								query_sent = query_sent + 'FILTER(NOT EXISTS {?y ns:%s ?sk0} || EXISTS {?y ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														  'FILTER(NOT EXISTS {?y ns:%s ?sk2} || EXISTS {?y ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											 % (from_path, from_path, yearData, to_path, to_path, yearData)

						"""升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）"""
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_2hop(topicEntityID, two_hop[1])
							if paths:
								# 暂时只要一个
								query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
									paths[0])

						# if F1 != 0:
						# 	is_use_entity_cons = False

						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
						elif order_DESC in query_e:
							paths = query_order_desc_2hop(topicEntityID, two_hop[1])
							# print(paths) #多个path
							if paths:
								desc_path = paths[0]
								for path in paths:
									if "end_date" in paths:
										desc_path = path
										break
								# 暂时只要一个
								query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
									desc_path)
						print(query_sent)
						if query_sent[-1] == '1' or query_sent[-1] == '}':
							preAnswer["Answers"] = get_answer_new(topicEntityID, query_sent)
						else:
							preAnswer["Answers"] = get_answer(topicEntityID, query_sent)
						preAnswer_list.append(preAnswer)
				else:
					preAnswer["Answers"] = []
					preAnswer_list.append(preAnswer)
	# with open(f"{args.output_path}/WebQSP_{name}_qid_preAnswer_k_{k}_kesai_{args.kesai}.json", 'w', encoding="utf-8") as f:
	with open(f"{args.output_path}/WebQSP_{name}_qid_preAnswer_onlygru.json", 'w', encoding="utf-8") as f:
		json.dump(preAnswer_list, f)
		print("预测答案已写入文件，正在计算F1...")

def Eval(args):
	# logger.add(join(args.output_path, 'F1_ACC_Recall.log'))
	# logger.add(join(args.output_path, f'F1_ACC_Recall_k_{args.K}.log'))
	w = open(f"{args.output_path}/Error_analysis.txt", "w")
	"""这里就是用每个问句的F1若比之前好就保存进去，然后加起来再算平均和其他的指标"""

	goldData = json.loads(open("data/webQSP/source_data/WebQSP.test.json", encoding="utf-8").read())
	predAnswers = json.loads(open(f"{args.output_path}/WebQSP_test_qid_preAnswer_onlygru.json", encoding="utf-8").read())
	# predAnswers = json.loads(open(f"{args.output_path}/WebQSP_test_qid_preAnswer_k_{args.K}_kesai_{args.kesai}.json", encoding="utf-8").read())

	PredAnswersById = {}

	for item in predAnswers:
		PredAnswersById[item["QuestionId"]] = item["Answers"]

	total = 0.0
	f1sum = 0.0
	recSum = 0.0
	precSum = 0.0
	numCorrect = 0
	hit = 0
	# 遍历每个问题
	for entry in goldData["Questions"]:

		skip = True
		for pidx in range(0, len(entry["Parses"])):
			np = entry["Parses"][pidx]
			if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"][
				"ParseQuality"] == "Complete":
				skip = False

		if (len(entry["Parses"]) == 0 or skip):
			continue

		total += 1

		id = entry["QuestionId"]

		if id not in PredAnswersById:
			print("The problem " + id + " is not in the prediction set")
			print("Continue to evaluate the other entries")
			w.write(id + "\n")
			continue

		# total += 1

		if len(entry["Parses"]) == 0:
			w.write(id + "\n")
			print("Empty parses in the gold set. Breaking!!")
			break

		predAnswers = PredAnswersById[id]
		bestf1 = -9999
		bestf1Rec = -9999
		bestf1Prec = -9999
		for pidx in range(0, len(entry["Parses"])):
			pidxAnswers = entry["Parses"][pidx]["Answers"]
			prec, rec, f1 = CalculatePRF1(pidxAnswers, predAnswers)
			if hit_n(pidxAnswers, predAnswers, 3):
				hit += 1
			if f1 == 0:
				w.write(id + "\n")
			if f1 > bestf1:
				bestf1 = f1
				bestf1Rec = rec
				bestf1Prec = prec

		f1sum += bestf1
		recSum += bestf1Rec
		precSum += bestf1Prec
		if bestf1 == 1.0:
			numCorrect += 1
	logger.info("Dataset: webQSP")
	logger.info(f"kesai: {args.kesai}")
	logger.info(f"k: {args.K}")
	logger.info(f"Number of questions: {int(total)}")
	logger.info("Average precision over questions: %.4f " % (precSum / total))
	logger.info("Average recall over questions: %.4f" % (recSum / total))
	logger.info("Average f1 over questions (accuracy): %.4f " % (f1sum / total))
	logger.info("F1 of average recall and average precision: %.4f " % (
			2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
	logger.info("True accuracy (ratio of questions answered exactly correctly): %.4f " % (numCorrect / total))
	logger.info("Hit@3: %.4f " % (hit / total))
	f1 = 2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)
	return f1

def webQSP_C3PM_test_beam_search_topk(name, args_train, args, k):
	# top_id_name_tname = return_type_topid()
	# print(f"测试集中有实体id的长度为：{len(top_id_name_tname)}")
	type_num = 0
	preEntity_num = 0
	print(f"正在评估超参数beam size:{k}, kesai:{args.kesai}的性能")
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	pt_path_train = join(args_train.output_path, "simcse.pt")
	model_train = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args_train.pretrain_model_path, pooling="cls", kesai=args_train.kesai, dropout=args_train.dropout).to(args_train.device)
	model_train.load_state_dict(torch.load(pt_path_train))


	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		preAnswer_list = []
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
				Answersinfo = i["Parses"][0]["Answers"]
				# 这里记录的是答案的id号
				Answers = []
				if Answersinfo:
					for answer in Answersinfo:
						Answers.append(answer["AnswerArgument"])
				"存储问题的实体类型 由于前期已经查询过 直接从文件获取 不再查库浪费时间 "
				# 获得主题实体类型名
				# topicEntityType = top_id_name_tname[QuestionId]["topicEntityTypeName"]
				# Sparql = i["Parses"][0]["Sparql"]
				"PosQueryGraph应该是正确的查询图"
				if i["Parses"][0]["InferentialChain"]:
					PosQueryGraph = i["Parses"][0]["InferentialChain"]

				preAnswer = {}
				preAnswer["QuestionId"] = QuestionId
				query_e = question.replace(mention, "<e>")
				# 返回的是问题的year list
				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
				one_hop_rels = get_1hop_p(topicEntityID)
				# 无约束
				if one_hop_rels:
					one_hop_nocons_data = []
					one_hop_rels_sortbycos = []
					is_use_entity_cons = True
					"""查找一跳无约束"""
					for one_hop_rel in one_hop_rels:
						cos = cosin_sim(query_e, one_hop_rel, args_train, model_train)
						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
					if one_hop_nocons_data:
						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
					else:
						preAnswer["Path"] = []
						preAnswer_list.append(preAnswer)
						continue
					# 这里暂且对top k个查询图进行后续约束加入
					if len(PosQueryGraph) == 1:
						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
						query_sent = one_hop[1]
						x_path = query_sent.replace("ns:", "").replace(" ?x .", "")

						preAnswer["Path"] = [x_path]
						print(preAnswer)
						preAnswer_list.append(preAnswer)

					elif len(PosQueryGraph) == 2:
						one_hop_rels_top_k = one_hop_rels_sortbycos[:k]
						# beam search找到的top k个路径，还未整体排序
						two_hop_all_candid = []
						# 两跳打了分之后
						two_hop_all_scored = []
						# 两跳打了分并排序
						for one_hop_in_two in one_hop_rels_top_k:
							one_hop_in_two = one_hop_in_two[1].replace(" ?x .", "")
							# 当前one_hop_in_two
							two_hop_rels = from_id_path_get_2hop_po(topicEntityID, one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
							if two_hop_rels:
								two_hop_nocons_data = []
								is_use_entity_cons = True
								for two_hop_rel in two_hop_rels:
									"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
									two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
									# 注意这里计算的是第二跳路径，没有两跳一起计算
									cos = cosin_sim(query_e, two_hop_rel, args_train, model_train)
									# 两条路径合并一起计算
									# cos = cosin_sim(query_e, two_hop)
									# 问题和当前one_hop_in_two第二跳路径的cos得分
									two_hop_nocons_data.append([query_e, two_hop, cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
								if two_hop_nocons_data:
									two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2], reverse=True)
									two_hop_all_candid += two_hop_rels_sortbycos[:k]
							else:
								continue
						if not two_hop_all_candid:
							preAnswer["Path"] = []
							preAnswer_list.append(preAnswer)
							continue
						else:
							# print(two_hop_all_candid)
							for two_hop in two_hop_all_candid:
								cos = cosin_sim_beam_search(query_e, two_hop[1], args, model_beam)
								two_hop_all_scored.append([two_hop[0], two_hop[1], cos])
							two_hop_all_scored_sortedbycos = sorted(two_hop_all_scored, key=lambda x: x[2], reverse=True)
						"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
						"""同样最后处理实体约束, 同样选前k个"""
						# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						two_hop = two_hop_all_scored_sortedbycos[0]  # [query_e, two_hop_rel, cos]
						query_sent = two_hop[1]
						# ?y后面
						y_path1 = query_sent.split(" ?y .?y ")[-1].replace("ns:", "").replace(" ?x .", "")
						# ?y前面
						y_path2 = query_sent.split(" ?y .?y ")[0].replace("ns:", "")
						preAnswer["Path"] = [y_path2, y_path1]
						print(preAnswer)
						preAnswer_list.append(preAnswer)


	with open(f"{args.output_path}/WebQSP_{name}_C3PM_core_path_predict.json", 'w', encoding="utf-8") as f:
		json.dump(preAnswer_list, f)
		print("预测答案已写入文件，正在计算F1...")

def C3PM_Eval(args):
	# logger.add(join(args.output_path, 'F1_ACC_Recall.log'))
	# logger.add(join(args.output_path, f'F1_ACC_Recall_k_{args.K}.log'))
	w = open(f"{args.output_path}/Error_analysis.txt", "w")
	"""这里就是用每个问句的F1若比之前好就保存进去，然后加起来再算平均和其他的指标"""

	goldData = json.loads(open("data/webQSP/source_data/WebQSP.test.json", encoding="utf-8").read())
	predAnswers = json.loads(open(f"{args.output_path}/WebQSP_test_C3PM_core_path_predict.json", encoding="utf-8").read())
	# predAnswers = json.loads(open(f"{args.output_path}/WebQSP_test_qid_preAnswer_k_{args.K}_kesai_{args.kesai}.json", encoding="utf-8").read())

	PredAnswersById = {}

	for item in predAnswers:
		PredAnswersById[item["QuestionId"]] = item["Path"]

	total = 0.0
	f1sum = 0.0
	recSum = 0.0
	precSum = 0.0
	numCorrect = 0
	hit = 0
	# 遍历每个问题
	for entry in goldData["Questions"]:

		skip = True
		for pidx in range(0, len(entry["Parses"])):
			np = entry["Parses"][pidx]
			if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"][
				"ParseQuality"] == "Complete":
				skip = False

		if (len(entry["Parses"]) == 0 or skip):
			continue

		total += 1

		id = entry["QuestionId"]

		if id not in PredAnswersById:
			print("The problem " + id + " is not in the prediction set")
			print("Continue to evaluate the other entries")
			w.write(id + "\n")
			continue

		# total += 1

		if len(entry["Parses"]) == 0:
			w.write(id + "\n")
			print("Empty parses in the gold set. Breaking!!")
			break

		predAnswers = PredAnswersById[id]
		bestf1 = -9999
		bestf1Rec = -9999
		bestf1Prec = -9999
		for pidx in range(0, len(entry["Parses"])):
			pidxAnswers = entry["Parses"][pidx]["InferentialChain"]
			prec, rec, f1 = CalculatePRF1_C3PM(pidxAnswers, predAnswers)
			if hit_n_complexqa(pidxAnswers, predAnswers, 1):
				hit += 1
			if f1 == 0:
				w.write(id + "\n")
			if f1 > bestf1:
				bestf1 = f1
				bestf1Rec = rec
				bestf1Prec = prec

		f1sum += bestf1
		recSum += bestf1Rec
		precSum += bestf1Prec
		if bestf1 == 1.0:
			numCorrect += 1
	logger.info("Dataset: webQSP")
	logger.info(f"kesai: {args.kesai}")
	logger.info(f"k: {args.K}")
	logger.info(f"Number of questions: {int(total)}")
	logger.info("Average precision over questions: %.4f " % (precSum / total))
	logger.info("Average recall over questions: %.4f" % (recSum / total))
	logger.info("Average f1 over questions (accuracy): %.4f " % (f1sum / total))
	logger.info("F1 of average recall and average precision: %.4f " % (
			2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
	logger.info("True accuracy (ratio of questions answered exactly correctly): %.4f " % (numCorrect / total))
	logger.info("Hit@1: %.4f " % (hit / total))
	f1 = 2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)
	return f1

# def complexQA_answer_predict(name, args, n, k):
# 	gender_male = ["dad", "father", "son", "brothers", "brother"]
# 	gender_female = ["mom", "daughter", "wife", "mother", "mum"]
# 	marry = ["husband", "married", "marry", "wife"]  # people.marriage.type_of_union
# 	order_ASC = "first"  # from start_date  ?y ns:sports.sports_team_roster.from ?sk0 . ORDER BY xsd:datetime(?sk0) LIMIT 1    #first name/ first wife /first language
# 	order_DESC = "last"  # last year ! = 2014
# 	# other_DESC = ["the most", "biggest", "largest", "predominant", "tallest", "major", "newly"]
#
# 	data_path = f"data/complexQA/source_data/compQ.{name}.json"
# 	with open(data_path, "r", encoding="utf-8") as f:
# 		data = json.load(f)
# 		# 没有答案的问题
# 		"""生成查询图"""
# 		# 存储问题id和答案
# 		preAnswer_list = []
# 		for i in tqdm(data):
# 			"""获取基本信息"""
# 			# 预测的跳数
# 			hop_num = 0
# 			QuestionId = i["questionId"]
# 			question = i["question"].lower().strip('?')
# 			mention = i["mention"]
# 			if mention is None:
# 				mention = ""
# 			query_e = question.replace(mention, "<e>")
# 			# query_e = query
# 			topicEntityID = i["TopicEntityMid"]
# 			# 是否有gold chain
# 			# infere_isnull = True
# 			try:
# 				InferentialChain = i["InferentialChain"]
# 			except Exception as e:
# 				print(e)
# 				InferentialChain = None
# 				hop_num = question_hop_num(question, n, k)
# 				pass
# 				# infere_isnull = False
# 				# continue
# 			"""生成训练数据"""
# 			if InferentialChain is not None or hop_num != 0:
# 				"PosQueryGraph应该是正确的查询图"
# 				if InferentialChain is not None:
# 					hop_num = len(InferentialChain)
# 				preAnswer = {}
# 				preAnswer["QuestionId"] = QuestionId
# 				query_sent = ""
# 				# 返回的是问题的year list
# 				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
# 				one_hop_rels = get_1hop_p(topicEntityID)
# 				# 无约束
# 				if one_hop_rels:
# 					one_hop_nocons_data = []
# 					is_use_entity_cons = True
# 					"""查找一跳无约束"""
# 					for one_hop_rel in one_hop_rels:
# 						cos = cosin_sim(query_e, one_hop_rel, args_train)
# 						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
# 						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
# 					if one_hop_nocons_data:
# 						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
# 					else:
# 						preAnswer["Answers"] = []
# 						preAnswer_list.append(preAnswer)
# 						continue
# 					# 这里暂且对top k个查询图进行后续约束加入
# 					if hop_num == 1:
# 						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
# 						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
# 						query_sent = one_hop[1] + "?x ns:type.object.name ?n ."
# 						"男性约束"
# 						if set(gender_male) & set(query_e.split()):
# 							# one_hop[1]格式"ns:" + one_hop_rel + " ?x ."
# 							query_sent = one_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
# 							is_use_entity_cons = False
#
# 						"女性约束"
# 						if set(gender_female) & set(query_e.split()):
# 							query_sent = one_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
# 							is_use_entity_cons = False
# 						"""实体约束"""
# 						if is_use_entity_cons:
# 							# 这里一跳的实体约束还可以再改进一下按照中间实体约束那样搞（后面再弄）
# 							one_hop_entity_constrains = from_id_path_get_2hop_po_oName(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
# 							# one_hop_entity_constrains = from_id_path_get_2hop_po_oName_new(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
# 							if one_hop_entity_constrains:
# 								# 这里可以对候选实体约束排个序选最高
# 								one_hop_entity_constrain = one_hop_entity_constrains[0]
# 								# [core_path,id.name] 这里是加完其他约束后根据情况再加实体约束
# 								query_sent = query_sent + f"?x ns:{one_hop_entity_constrain[0]} ns:{one_hop_entity_constrain[1]} ."
# 						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
# 						if yearCons:  # 如果问题中有年份的时间约束
# 							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
# 							# 至于约束的名称则需要在库里面查询 （ from ,to ）
# 							from_path, to_path = query_1hop_p_from_to(topicEntityID, one_hop[1], yearData)  # 选出成对的路径
# 							if from_path and to_path:
# 								query_sent = query_sent + 'FILTER(NOT EXISTS {?x ns:%s ?sk0} || EXISTS {?x ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
# 														  'FILTER(NOT EXISTS {?x ns:%s ?sk2} || EXISTS {?x ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
# 											 % (from_path, from_path, yearData, to_path, to_path, yearData)
# 						# 升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）
# 						if order_ASC in query_e:
# 							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
# 							paths = query_order_asc_1hop(topicEntityID, one_hop[1])
# 							if paths:
# 								# 暂时只要一个
# 								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
# 									paths[0])
# 						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
# 						elif order_DESC in query_e:
# 							paths = query_order_desc_1hop(topicEntityID, one_hop[1])
# 							# print(paths) #多个path
# 							if paths:
# 								if "end_date" in paths:
# 									desc_path = "end_date"
# 								elif "start_date" in paths:
# 									desc_path = "start_date"
# 								else:
# 									desc_path = paths[0]
# 								# 暂时只要一个
# 								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
# 									desc_path)
# 						if query_sent[-1] == '1' or query_sent[-1] == '}':
# 							preAnswer["Answers"] = get_answer_name_new(topicEntityID, query_sent)
# 						else:
# 							preAnswer["Answers"] = get_answer_name(topicEntityID, query_sent)
# 						preAnswer_list.append(preAnswer)
# 					elif hop_num == 2:
# 						one_hop_in_two = one_hop_rels_sortbycos[0][1].replace(" ?x .", "")
# 						two_hop_rels = from_id_path_get_2hop_po(topicEntityID,
# 																one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
# 						if two_hop_rels:
# 							two_hop_nocons_data = []
# 							is_use_entity_cons = True
# 							for two_hop_rel in two_hop_rels:
# 								"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
# 								two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
# 								# 注意这里计算的是第二跳路径，没有两跳一起计算
# 								cos = cosin_sim(query_e, two_hop_rel, args_train)
# 								# 两跳合并一起计算
# 								# cos = cosin_sim(query_e, two_hop, args_train)
# 								two_hop_nocons_data.append([query_e, two_hop, cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
# 							if two_hop_nocons_data:
# 								two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2], reverse=True)
# 							else:
# 								preAnswer["Answers"] = []
# 								preAnswer_list.append(preAnswer)
# 								continue
# 							"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
# 							"""同样最后处理实体约束, 同样选前k个"""
# 							# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
# 							two_hop = two_hop_rels_sortbycos[0]  # [query_e, two_hop_rel, cos, F1]
# 							query_sent = two_hop[1] + "?x ns:type.object.name ?n ."
#
# 							"""男性约束"""
# 							if set(gender_male) & set(query_e.split()):
# 								query_sent = two_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
# 								is_use_entity_cons = False
#
# 							"""女性约束"""
# 							if set(gender_female) & set(query_e.split()):
# 								query_sent = two_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
# 								is_use_entity_cons = False
# 							"是否结婚的约束"
# 							if set(marry) & set(query_e.split()):
# 								query_sent = two_hop[1] + "?x ns:type.object.name ?n ." + f"?y ns:people.marriage.type_of_union ns:m.04ztj ."  # 婚姻 约束在y上 是否都有时间限制
# 							"""实体约束"""
# 							if is_use_entity_cons:
# 								"""处理约束 实体约束 约束加载第二跳的实体上"""  # 实体存在问题中 查询出第二跳的实体名称及路径
# 								two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x(question, topicEntityID, two_hop[1])
# 								# two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x_new(question, topicEntityID, two_hop[1])
# 								"两跳实体约束（加在x上的）"
# 								if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
# 									two_hop_entity_constrain = two_hop_entity_constrains_x[0]
# 									# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
# 									query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
# 									# for two_hop_entity_constrain in two_hop_entity_constrains_x:
# 									# 	# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
# 									# 	query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
# 								# 在y上的实体约束
# 								"两跳实体约束（加在y上的）"
# 								two_hop_entity_constrains_y = from_id_path_get_3hop_po_oName_y(question,
# 																							   topicEntityID,
# 																							   two_hop[1])
# 								if two_hop_entity_constrains_y:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
# 									two_hop_entity_constrain = two_hop_entity_constrains_y
# 									# two_hop_entity_constrain:[path, id, name]
# 									query_sent = query_sent + f"?y ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
#
# 							"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
# 							if yearCons:  # 如果问题中有年份的时间约束
# 								yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
# 								# 至于约束的名称则需要在库里面查询 （ from ,to ）
# 								from_path, to_path = query_2hop_p_from_to(topicEntityID, two_hop[1],
# 																		  yearData)  # 选出成对的路径
# 								if from_path and to_path:
# 									# 数据集中gender之后没有year约束但是有DESC约束
# 									query_sent = query_sent + 'FILTER(NOT EXISTS {?y ns:%s ?sk0} || EXISTS {?y ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
# 															  'FILTER(NOT EXISTS {?y ns:%s ?sk2} || EXISTS {?y ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
# 												 % (from_path, from_path, yearData, to_path, to_path, yearData)
#
# 							"""升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）"""
# 							if order_ASC in query_e:
# 								# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
# 								paths = query_order_asc_2hop(topicEntityID, two_hop[1])
# 								if paths:
# 									# 暂时只要一个
# 									query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
# 										paths[0])
#
# 							# if F1 != 0:
# 							# 	is_use_entity_cons = False
#
# 							# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
# 							elif order_DESC in query_e:
# 								paths = query_order_desc_2hop(topicEntityID, two_hop[1])
# 								# print(paths) #多个path
# 								if paths:
# 									desc_path = paths[0]
# 									for path in paths:
# 										if "end_date" in paths:
# 											desc_path = path
# 											break
# 									# 暂时只要一个
# 									query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (desc_path)
# 							print(query_sent)
# 							if query_sent[-1] == '1' or query_sent[-1] == '}':
# 								preAnswer["Answers"] = get_answer_name_new(topicEntityID, query_sent)
# 							else:
# 								preAnswer["Answers"] = get_answer_name(topicEntityID, query_sent)
# 							preAnswer_list.append(preAnswer)
# 					else:
# 						preAnswer["Answers"] = []
# 						preAnswer_list.append(preAnswer)
# 		with open(f"{args.output_path}/{args.dataset}_{name}_preAnswer_all.json", 'w', encoding="utf-8") as f:
# 			json.dump(preAnswer_list, f)
# 			print("预测答案已写入文件，正在计算F1...")

def complexQA_answer_predict_beam_search_topk(name, args_train, args, n, m, k):


	print(f"正在评估超参数beam size:{k}, kesai:{args.kesai}的性能")
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	pt_path_train = join(args_train.output_path, "simcse.pt")
	model_train = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args_train.pretrain_model_path, pooling="cls", kesai=args_train.kesai, dropout=args_train.dropout).to(args_train.device)
	model_train.load_state_dict(torch.load(pt_path_train))
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	gender_male = ["dad", "father", "son", "brothers", "brother"]
	gender_female = ["mom", "daughter", "wife", "mother", "mum"]
	marry = ["husband", "married", "marry", "wife"]  # people.marriage.type_of_union
	order_ASC = "first"  # from start_date  ?y ns:sports.sports_team_roster.from ?sk0 . ORDER BY xsd:datetime(?sk0) LIMIT 1    #first name/ first wife /first language
	order_DESC = "last"  # last year ! = 2014
	# other_DESC = ["the most", "biggest", "largest", "predominant", "tallest", "major", "newly"]

	data_path = f"data/complexQA/source_data/compQ.{name}.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		preAnswer_list = []
		for i in tqdm(data):
			"""获取基本信息"""
			# 预测的跳数
			hop_num = 0
			QuestionId = i["questionId"]
			question = i["question"].lower().strip('?')
			mention = i["mention"]
			if mention is None:
				mention = ""
			query_e = question.replace(mention, "<e>")
			# query_e = query
			topicEntityID = i["TopicEntityMid"]
			# 是否有gold chain
			# infere_isnull = True
			try:
				InferentialChain = i["InferentialChain"]
			except Exception as e:
				print(e)
				InferentialChain = None
				hop_num = question_hop_num(question, n, m)
				pass
				# infere_isnull = False
				# continue
			"""生成训练数据"""
			if InferentialChain is not None or hop_num != 0:
				"PosQueryGraph应该是正确的查询图"
				if InferentialChain is not None:
					hop_num = len(InferentialChain)
				preAnswer = {}
				preAnswer["QuestionId"] = QuestionId
				query_sent = ""
				# 返回的是问题的year list
				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
				one_hop_rels = get_1hop_p(topicEntityID)
				# 无约束
				if one_hop_rels:
					one_hop_nocons_data = []
					is_use_entity_cons = True
					"""查找一跳无约束"""
					for one_hop_rel in one_hop_rels:
						cos = cosin_sim(query_e, one_hop_rel, args_train, model_train)
						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
					if one_hop_nocons_data:
						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
					else:
						preAnswer["Answers"] = []
						preAnswer_list.append(preAnswer)
						continue
					# 这里暂且对top k个查询图进行后续约束加入
					if hop_num == 1:
						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
						query_sent = one_hop[1] + "?x ns:type.object.name ?n ."

						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
						"男性约束"
						if set(gender_male) & set(query_e.split()):
							# one_hop[1]格式"ns:" + one_hop_rel + " ?x ."
							query_sent = one_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							is_use_entity_cons = False

						"女性约束"
						if set(gender_female) & set(query_e.split()):
							query_sent = one_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							is_use_entity_cons = False
						"""实体约束"""
						if is_use_entity_cons:
							# 这里一跳的实体约束还可以再改进一下按照中间实体约束那样搞（后面再弄）
							one_hop_entity_constrains = from_id_path_get_2hop_po_oName(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
							# one_hop_entity_constrains = from_id_path_get_2hop_po_oName_new(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
							if one_hop_entity_constrains:
								# 这里可以对候选实体约束排个序选最高
								one_hop_entity_constrain = one_hop_entity_constrains[0]
								# [core_path,id.name] 这里是加完其他约束后根据情况再加实体约束
								query_sent = query_sent + f"?x ns:{one_hop_entity_constrain[0]} ns:{one_hop_entity_constrain[1]} ."
						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_1hop_p_from_to(topicEntityID, one_hop[1], yearData)  # 选出成对的路径
							if from_path and to_path:
								query_sent = query_sent + 'FILTER(NOT EXISTS {?x ns:%s ?sk0} || EXISTS {?x ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														  'FILTER(NOT EXISTS {?x ns:%s ?sk2} || EXISTS {?x ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											 % (from_path, from_path, yearData, to_path, to_path, yearData)
						# 升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_1hop(topicEntityID, one_hop[1])
							if paths:
								# 暂时只要一个
								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
									paths[0])
						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
						elif order_DESC in query_e:
							paths = query_order_desc_1hop(topicEntityID, one_hop[1])
							# print(paths) #多个path
							if paths:
								if "end_date" in paths:
									desc_path = "end_date"
								elif "start_date" in paths:
									desc_path = "start_date"
								else:
									desc_path = paths[0]
								# 暂时只要一个
								query_sent = query_sent + "?x ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
									desc_path)
						if query_sent[-1] == '1' or query_sent[-1] == '}':
							preAnswer["Answers"] = get_answer_name_new(topicEntityID, query_sent)
						else:
							preAnswer["Answers"] = get_answer_name(topicEntityID, query_sent)
						preAnswer_list.append(preAnswer)
					elif hop_num == 2:
						one_hop_rels_top_k = one_hop_rels_sortbycos[:k]
						# beam search找到的top k个路径，还未整体排序
						two_hop_all_candid = []
						# 两跳打了分之后
						two_hop_all_scored = []
						# 两跳打了分并排序
						two_hop_all_scored_sortedbycos = []
						for one_hop_in_two in one_hop_rels_top_k:
							one_hop_in_two = one_hop_in_two[1].replace(" ?x .", "")
							# 当前one_hop_in_two
							two_hop_rels = from_id_path_get_2hop_po(topicEntityID,
																	one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
							if two_hop_rels:
								two_hop_nocons_data = []
								is_use_entity_cons = True
								for two_hop_rel in two_hop_rels:
									"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
									two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
									# 注意这里计算的是第二跳路径，没有两跳一起计算
									cos = cosin_sim(query_e, two_hop_rel, args_train, model_train)
									# 两条路径合并一起计算
									# cos = cosin_sim(query_e, two_hop)
									# 问题和当前one_hop_in_two第二跳路径的cos得分
									two_hop_nocons_data.append([query_e, two_hop,
																cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
								if two_hop_nocons_data:
									two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2],
																	reverse=True)
									two_hop_all_candid += two_hop_rels_sortbycos[:k]
							else:
								continue
						if not two_hop_all_candid:
							preAnswer["Answers"] = []
							preAnswer_list.append(preAnswer)
							continue
						else:
							for two_hop in two_hop_all_candid:
								cos = cosin_sim_beam_search(query_e, two_hop[1], args, model_beam)
								two_hop_all_scored.append([two_hop[0], two_hop[1], cos])
							two_hop_all_scored_sortedbycos = sorted(two_hop_all_scored, key=lambda x: x[2], reverse=True)
						"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
						"""同样最后处理实体约束, 同样选前k个"""
						# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						two_hop = two_hop_all_scored_sortedbycos[0]
						query_sent = two_hop[1] + "?x ns:type.object.name ?n ."
						"""男性约束"""
						if set(gender_male) & set(query_e.split()):
							query_sent = two_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							is_use_entity_cons = False

						"""女性约束"""
						if set(gender_female) & set(query_e.split()):
							query_sent = two_hop[1] + "?x ns:type.object.name ?n ." + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							is_use_entity_cons = False
						"是否结婚的约束"
						if set(marry) & set(query_e.split()):
							query_sent = two_hop[1] + "?x ns:type.object.name ?n ." + f"?y ns:people.marriage.type_of_union ns:m.04ztj ."  # 婚姻 约束在y上 是否都有时间限制
						"""实体约束"""
						if is_use_entity_cons:
							"""处理约束 实体约束 约束加载第二跳的实体上"""  # 实体存在问题中 查询出第二跳的实体名称及路径
							two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x(question, topicEntityID, two_hop[1])
							# two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x_new(question, topicEntityID, two_hop[1])
							"两跳实体约束（加在x上的）"
							if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
								two_hop_entity_constrain = two_hop_entity_constrains_x[0]
								# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
								query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
							# for two_hop_entity_constrain in two_hop_entity_constrains_x:
							# 	# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
							# 	query_sent = query_sent + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
							# 在y上的实体约束
							"两跳实体约束（加在y上的）"
							two_hop_entity_constrains_y = from_id_path_get_3hop_po_oName_y(question,
																						   topicEntityID,
																						   two_hop[1])
							if two_hop_entity_constrains_y:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
								two_hop_entity_constrain = two_hop_entity_constrains_y
								# two_hop_entity_constrain:[path, id, name]
								query_sent = query_sent + f"?y ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."

						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_2hop_p_from_to(topicEntityID, two_hop[1],
																	  yearData)  # 选出成对的路径
							if from_path and to_path:
								# 数据集中gender之后没有year约束但是有DESC约束
								query_sent = query_sent + 'FILTER(NOT EXISTS {?y ns:%s ?sk0} || EXISTS {?y ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														  'FILTER(NOT EXISTS {?y ns:%s ?sk2} || EXISTS {?y ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											 % (from_path, from_path, yearData, to_path, to_path, yearData)

						"""升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）"""
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_2hop(topicEntityID, two_hop[1])
							if paths:
								# 暂时只要一个
								query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (
									paths[0])

						# if F1 != 0:
						# 	is_use_entity_cons = False

						# 降序约束（较多，但不好分别，目前只能用简单的关键字"last"来判别）
						elif order_DESC in query_e:
							paths = query_order_desc_2hop(topicEntityID, two_hop[1])
							# print(paths) #多个path
							if paths:
								desc_path = paths[0]
								for path in paths:
									if "end_date" in paths:
										desc_path = path
										break
								# 暂时只要一个
								query_sent = query_sent + "?y ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (desc_path)
						print(query_sent)
						if query_sent[-1] == '1' or query_sent[-1] == '}':
							preAnswer["Answers"] = get_answer_name_new(topicEntityID, query_sent)
						else:
							preAnswer["Answers"] = get_answer_name(topicEntityID, query_sent)
						preAnswer_list.append(preAnswer)
				else:
					preAnswer["Answers"] = []
					preAnswer_list.append(preAnswer)
	# with open(f"{args.output_path}/{args.dataset}_{name}_preAnswer_k_{k}_kesai_{args.kesai}.json", 'w', encoding="utf-8") as f:
	with open(f"{args.output_path}/{args.dataset}_{name}_preAnswer_onlycnn.json", 'w', encoding="utf-8") as f:
		json.dump(preAnswer_list, f)
		print("预测答案已写入文件，正在计算F1...")

def complexQA_C3PM_test_beam_search_topk(name, args_train, args, n, m, k):


	# print(f"正在评估超参数beam size:{k}, kesai:{args.kesai}的性能")
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	pt_path_train = join(args_train.output_path, "simcse.pt")
	model_train = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args_train.pretrain_model_path, pooling="cls", kesai=args_train.kesai, dropout=args_train.dropout).to(args_train.device)
	model_train.load_state_dict(torch.load(pt_path_train))
	# 加载模型
	pt_path_beam = join(args.output_path, "simcse.pt")
	model_beam = SimcseModel_T5_GRU_CNN_Attention(pretrained_model=args.pretrain_model_path, pooling="cls", kesai=args.kesai, dropout=args.dropout).to(args.device)
	model_beam.load_state_dict(torch.load(pt_path_beam))

	data_path = f"data/complexQA/source_data/compQ.{name}.json"
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		"""生成查询图"""
		# 存储问题id和答案
		preAnswer_list = []
		for i in tqdm(data):
			"""获取基本信息"""
			# 预测的跳数
			hop_num = 0
			QuestionId = i["questionId"]
			question = i["question"].lower().strip('?')
			mention = i["mention"]
			if mention is None:
				mention = ""
			query_e = question.replace(mention, "<e>")
			# query_e = query
			topicEntityID = i["TopicEntityMid"]
			# 是否有gold chain
			# infere_isnull = True
			try:
				InferentialChain = i["InferentialChain"]
			except Exception as e:
				print(e)
				InferentialChain = None
				hop_num = question_hop_num(question, n, m)
				pass
				# infere_isnull = False
				# continue
			"""生成训练数据"""
			if InferentialChain is not None or hop_num != 0:
				"PosQueryGraph应该是正确的查询图"
				if InferentialChain is not None:
					hop_num = len(InferentialChain)
				preAnswer = {}
				preAnswer["QuestionId"] = QuestionId
				query_sent = ""
				# 返回的是问题的year list
				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
				one_hop_rels = get_1hop_p(topicEntityID)
				# 无约束
				if one_hop_rels:
					one_hop_nocons_data = []
					is_use_entity_cons = True
					"""查找一跳无约束"""
					for one_hop_rel in one_hop_rels:
						cos = cosin_sim(query_e, one_hop_rel, args_train, model_train)
						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
						one_hop_nocons_data.append([query_e, one_hop_rel, cos])
					if one_hop_nocons_data:
						one_hop_rels_sortbycos = sorted(one_hop_nocons_data, key=lambda x: x[2], reverse=True)
					else:
						preAnswer["Path"] = []
						preAnswer_list.append(preAnswer)
						continue
					# 这里暂且对top k个查询图进行后续约束加入
					if hop_num == 1:
						one_hop = one_hop_rels_sortbycos[0]  # [query_e, one_hop_rel, cos]
						query_sent = one_hop[1]
						x_path = query_sent.replace("ns:", "").replace(" ?x .", "")

						preAnswer["Path"] = [x_path]
						print(preAnswer)
						preAnswer_list.append(preAnswer)
					elif hop_num == 2:
						one_hop_rels_top_k = one_hop_rels_sortbycos[:k]
						# beam search找到的top k个路径，还未整体排序
						two_hop_all_candid = []
						# 两跳打了分之后
						two_hop_all_scored = []
						# 两跳打了分并排序
						two_hop_all_scored_sortedbycos = []
						for one_hop_in_two in one_hop_rels_top_k:
							one_hop_in_two = one_hop_in_two[1].replace(" ?x .", "")
							# 当前one_hop_in_two
							two_hop_rels = from_id_path_get_2hop_po(topicEntityID,
																	one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
							if two_hop_rels:
								two_hop_nocons_data = []
								is_use_entity_cons = True
								for two_hop_rel in two_hop_rels:
									"""这里先第二跳单独计算，后面再试试两跳一起去计算相似度，看看哪个效率高"""
									two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
									# 注意这里计算的是第二跳路径，没有两跳一起计算
									cos = cosin_sim(query_e, two_hop_rel, args_train, model_train)
									# 两条路径合并一起计算
									# cos = cosin_sim(query_e, two_hop)
									# 问题和当前one_hop_in_two第二跳路径的cos得分
									two_hop_nocons_data.append([query_e, two_hop,
																cos])  # two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
								if two_hop_nocons_data:
									two_hop_rels_sortbycos = sorted(two_hop_nocons_data, key=lambda x: x[2],
																	reverse=True)
									two_hop_all_candid += two_hop_rels_sortbycos[:k]
							else:
								continue
						if not two_hop_all_candid:
							preAnswer["Path"] = []
							preAnswer_list.append(preAnswer)
							continue
						else:
							for two_hop in two_hop_all_candid:
								cos = cosin_sim_beam_search(query_e, two_hop[1], args, model_beam)
								two_hop_all_scored.append([two_hop[0], two_hop[1], cos])
							two_hop_all_scored_sortedbycos = sorted(two_hop_all_scored, key=lambda x: x[2], reverse=True)
						"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
						"""同样最后处理实体约束, 同样选前k个"""
						# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						two_hop = two_hop_all_scored_sortedbycos[0]
						query_sent = two_hop[1]
						# ?y后面
						y_path1 = query_sent.split(" ?y .?y ")[-1].replace("ns:", "").replace(" ?x .", "")
						# ?y前面
						y_path2 = query_sent.split(" ?y .?y ")[0].replace("ns:", "")
						preAnswer["Path"] = [y_path2, y_path1]
						print(preAnswer)
						preAnswer_list.append(preAnswer)

	with open(f"{args.output_path}/{args.dataset}_{name}_C3PM_core_path_predict.json", 'w', encoding="utf-8") as f:
		json.dump(preAnswer_list, f)
		print("预测答案已写入文件，正在计算F1...")


def CQ_Eval(args):

	# logger.add(join(args.output_path, 'F1_ACC_Recall.log'))
	"""这里就是用每个问句的F1若比之前好就保存进去，然后加起来再算平均和其他的指标"""


	goldData = json.loads(open("data/complexQA/source_data/compQ.test.json", encoding="utf-8").read())
	predAnswers = json.loads(open(f"{args.output_path}/comQA_test_preAnswer_onlycnn.json", encoding="utf-8").read())

	PredAnswersById = {}

	for item in predAnswers:
		PredAnswersById[item["QuestionId"]] = item["Answers"]

	total = 0.0
	f1sum = 0.0
	recSum = 0.0
	precSum = 0.0
	numCorrect = 0
	hit = 0
	for entry in goldData:

		# 原始数据集里的问题id（包含了所有）
		id = entry["questionId"]
		total += 1
		if id not in PredAnswersById:
			# print("The problem " + id + " is not in the prediction set")
			# print("Continue to evaluate the other entries")
			continue


		predAnswers = PredAnswersById[id]

		bestf1 = -9999
		bestf1Rec = -9999
		bestf1Prec = -9999

		pidxAnswers = entry["answer"]
		prec, rec, f1 = CQ_CalculatePRF1(pidxAnswers, predAnswers)
		if hit_n_complexqa(pidxAnswers, predAnswers, 3):
			hit += 1
		if f1 > bestf1:
			bestf1 = f1
			bestf1Rec = rec
			bestf1Prec = prec

		f1sum += bestf1
		recSum += bestf1Rec
		precSum += bestf1Prec
		if bestf1 == 1.0:
			numCorrect += 1

	logger.info(f"Dataset: comQA")
	logger.info(f"Number of questions: {int(total)}")
	logger.info(f"kesai: {args.kesai}")
	logger.info(f"k: {args.K}")
	logger.info("Average precision over questions: %.4f " % (precSum / total))
	logger.info("Average recall over questions: %.4f" % (recSum / total))
	logger.info("Average f1 over questions (accuracy): %.4f " % (f1sum / total))
	logger.info("F1 of average recall and average precision: %.4f " % (
			2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
	logger.info("True accuracy (ratio of questions answered exactly correctly): %.4f " % (numCorrect / total))
	logger.info("Hit@3: %.4f " % (hit / total))
	return 2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)

def CQ_Eval_C3PM(args):

	# logger.add(join(args.output_path, 'F1_ACC_Recall.log'))
	"""这里就是用每个问句的F1若比之前好就保存进去，然后加起来再算平均和其他的指标"""


	goldData = json.loads(open("data/complexQA/source_data/compQ.test.json", encoding="utf-8").read())
	predAnswers = json.loads(open(f"{args.output_path}/comQA_test_C3PM_core_path_predict.json", encoding="utf-8").read())

	PredAnswersById = {}

	for item in predAnswers:
		PredAnswersById[item["QuestionId"]] = item["Path"]

	total = 0.0
	f1sum = 0.0
	recSum = 0.0
	precSum = 0.0
	numCorrect = 0
	hit = 0
	for entry in goldData:

		# 原始数据集里的问题id（包含了所有）
		id = entry["questionId"]
		if id in PredAnswersById:
			# print("The problem " + id + " is not in the prediction set")
			# print("Continue to evaluate the other entries")
			try:
				InferentialChain = entry["InferentialChain"]
			except Exception as e:
				print(e)
				continue
			total += 1


			predAnswers = PredAnswersById[id]

			bestf1 = -9999
			bestf1Rec = -9999
			bestf1Prec = -9999

			pidxAnswers = entry["InferentialChain"]
			prec, rec, f1 = CQ_CalculatePRF1(pidxAnswers, predAnswers)
			if hit_n_complexqa(pidxAnswers, predAnswers, 1):
				hit += 1
			if f1 > bestf1:
				bestf1 = f1
				bestf1Rec = rec
				bestf1Prec = prec

			f1sum += bestf1
			recSum += bestf1Rec
			precSum += bestf1Prec
			if bestf1 == 1.0:
				numCorrect += 1

	logger.info(f"Dataset: comQA")
	logger.info(f"Number of questions: {int(total)}")
	logger.info(f"kesai: {args.kesai}")
	logger.info(f"k: {args.K}")
	logger.info("Average precision over questions: %.4f " % (precSum / total))
	logger.info("Average recall over questions: %.4f" % (recSum / total))
	logger.info("Average f1 over questions (accuracy): %.4f " % (f1sum / total))
	logger.info("F1 of average recall and average precision: %.4f " % (
			2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
	logger.info("True accuracy (ratio of questions answered exactly correctly): %.4f " % (numCorrect / total))
	logger.info("Hit@1: %.4f " % (hit / total))
	return 2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)


if __name__ == '__main__':
	# for i in [1,2,3,4,5]:
	#     create("test", True, i, 0.2)


	# args_train = parse_args_train()
	# args = parse_args()

	# webQSP_answer_predict("test", args, 15, 9)
	# webQSP_answer_predict_beam_search_topk("test", args, 5)
	# args.kesai = 0.5
	#
	# webQSP_answer_predict_beam_search_topk_only_constrained("test", args_train, args, 3)
	#
	# logger.add(join(args.output_path, f'F1_ACC_Recall_uncons_two_hop.log'))
	#
	# Eval(args)


	# complexQA_answer_predict('test', args)
	# complexQA_answer_predict_beam_search_topk("test", args, 10, 7, 5)
	# CQ_Eval(args)

	args_train = parse_args_train()
	args = parse_args()
	logger.add(join(args.output_path, f'F1_onlycnn.log'))
	complexQA_answer_predict_beam_search_topk("test", args_train, args, 10, 7, args.K)
	CQ_Eval(args)
	# webQSP_answer_predict_beam_search_topk_only_constrained("test", args_train, args, args.K)
	# Eval(args)

	# C3PM子模块评估
	# webQSP_C3PM_test_beam_search_topk("test", args_train, args, 3)
	# C3PM_Eval(args)

	# complexQA_C3PM_test_beam_search_topk("test", args_train, args, 10, 7, 3)
	# CQ_Eval_C3PM(args)



	# kesai超参数验证
	# # for i in [2]:
	# for i in [3, 4, 5]:
	# 	args.K = i
	# 	args_train.K = i
	# 	logger.add(join(args.output_path, f'F1_ACC_Recall_k_{args.K}.log'))
	# 	for j in [0.1, 0.3, 0.5, 0.7, 0.9]:
	# 	# for j in [0.5, 0.7, 0.9]:
	# 		args.kesai = j
	# 		args_train.kesai = j
	# 		# webQSP_answer_predict_beam_search_topk("test", args_train, args, args.K)
	# 		complexQA_answer_predict_beam_search_topk("test", args_train, args, 10, 7, args.K)
	# 		# Eval(args)
	# 		CQ_Eval(args)



# print(CalculatePRF1([], [])[-1])
# s1 = "what does jamaican people speak"
# s2 = ["location.country.languages_spoken",
# 	  "location.country.administrative_divisions", "location.country.calling_code",
# 	  "location.country.fifa_code", "location.country.official_language",
# 	  "book.book_subject.works"]
# print(Cal_q_path_score(s1, s2))

# baseline
# 2023-07-18 15:30:07.348 | INFO     | __main__:Eval:403 - Number of questions:
# 2023-07-18 15:30:07.348 | INFO     | __main__:Eval:404 - Average precision over questions: 0.7675
# 2023-07-18 15:30:07.349 | INFO     | __main__:Eval:405 - Average recall over questions: 0.7808
# 2023-07-18 15:30:07.349 | INFO     | __main__:Eval:406 - Average f1 over questions (accuracy): 0.7356
# 2023-07-18 15:30:07.349 | INFO     | __main__:Eval:408 - F1 of average recall and average precision: 0.7741
# 2023-07-18 15:30:07.349 | INFO     | __main__:Eval:409 - True accuracy (ratio of questions answered exactly correctly): 0.6681
