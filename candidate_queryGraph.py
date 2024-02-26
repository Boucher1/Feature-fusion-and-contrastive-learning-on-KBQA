#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/7
# @Author  : hehl
# @Software: PyCharm
# @File    : candidate_queryGraph.py
import csv
import json
from os.path import join

from tqdm import tqdm

from utils import cosin_sim
from loguru import logger
from query_virtuoso import *
from train import parse_args_train
from utils import cosin_sim
import warnings
"""该文件只用作生成查询图查看查询情况，不用做实际测试"""

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


# 这里的计算F1方法是用答案实体的id号来匹配的
def CalculatePRF1(predlist, glist):
	if len(glist) == 0:
		if len(predlist) == 0:
			return 1.0, 1.0, 1.0  # consider it 'correct' when there is no labeled answer, and also no predicted answer
		else:
			return 0.0, 1.0, 0.0  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
	elif len(predlist) == 0:
		return 1.0, 0.0, 0.0  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
	else:
		tp = 0.0  # numerical trick
		fp = 0.0
		fn = 0.0

		for gentry in predlist:
			if FindInList(gentry, glist):
				tp += 1
			else:
				fp += 1
		for pentry in glist:
			if not FindInList(pentry, predlist):
				fn += 1
		if tp == 0:
			return 0.0, 0.0, 0.0
		else:
			precision = tp / (tp + fp)
			recall = tp / (tp + fn)

			f1 = (2 * precision * recall) / (precision + recall)
		return precision, recall, f1



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


# 获取测试集问题的主题实体类型、id和名字
def return_type_topid():
	with open("data/webQSP/test_qid_top_Ename_Eid_Etype.json", 'r', encoding="utf-8") as f:
		data = json.load(f)
	return data


"""这里生成的查询图是按照步骤（主题实体-->主路径-->约束）来的，先试试效果，后期再考虑从文件获取打好分的主路径，只需要选择最高的主路径再加约束就去查库计算F1"""
"""或者还是按照步骤来，只需要调一个约束选择的包来选择约束"""
"""训练集有些是没有InferentialChain需要跳过"""


def create(name, k=5):
	# top_id_name_tname = return_type_topid()
	# print(f"测试集中有实体id的长度为：{len(top_id_name_tname)}")
	type_num = 0
	preEntity_num = 0
	"""生成候选查询图"""
	logger.info("开始生成候选查询图.....")

	gender_male = ["dad", "father", "son", "brothers", "brother"]
	gender_female = ["mom", "daughter", "wife", "mother", "mum"]
	marry = ["husband", "married", "marry", "wife"]  # people.marriage.type_of_union
	order_ASC = "first"  # from start_date  ?y ns:sports.sports_team_roster.from ?sk0 . ORDER BY xsd:datetime(?sk0) LIMIT 1    #first name/ first wife /first language
	order_DESC = "last"  # last year ! = 2014
	other_DESC = ["the most", "biggest", "largest", "predominant", "tallest", "major", "newly"]

	data_path = f"data/webQSP/source_data/WebQSP.{name}.json"
	f1 = open(f"data/webQSP/mask_data/result/{name}_Nocon_score_F1.csv", 'a', newline='')
	f2 = open(f"data/webQSP/mask_data/result/{name}_con_score_F1.csv", 'a', newline='')
	f1_writer = csv.writer(f1, delimiter='\t')
	f2_writer = csv.writer(f2, delimiter='\t')
	f1_writer.writerow(['sent0', 'sent1', 'score', 'F1'])
	f2_writer.writerow(['sent0', 'sent1', 'score', 'F1'])
	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# 没有答案的问题
		no_answer = 0
		# 计数器，用于统计能一跳查到答案的但是潜在路径为两跳的问题
		twohop_but_onehop = 0
		twohop_but_onehop_cons = 0
		logger.info(f"数据集{name}")
		"""生成查询图"""
		for i in tqdm(data["Questions"][2788:]):
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
				else:
					no_answer += 1
					logger.info(f"数据集{name}问题{QuestionId}没有答案，跳过")
				"存储问题的实体类型 由于前期已经查询过 直接从文件获取 不再查库浪费时间 "
				# 获得主题实体类型名
				# topicEntityType = top_id_name_tname[QuestionId]["topicEntityTypeName"]
				# Sparql = i["Parses"][0]["Sparql"]
				logger.info(
					f"问题id{QuestionId}, 问题{question}, 主题实体id{topicEntityID}, 主题实体名{topicEntityName}")
				"PosQueryGraph应该是正确的查询图"
				if i["Parses"][0]["InferentialChain"]:
					PosQueryGraph = i["Parses"][0]["InferentialChain"]
				else:
					logger.info("该问题没有InferentialChain，跳过")
					continue
				query_e = question.replace(mention, "<e>")
				# 返回的是问题的year list
				yearCons = find_year(query_e)  # 查找问题中有没有明确的时间约束 2012,2009....
				"""前期这里用的所有的查询图，后期模型训练好后采用前topk个减小数量"""
				one_hop_rels = get_1hop_p(topicEntityID)
				"""这里就先试试测试集的cosin和F1从高到底排序，后面生成训练数据时再把正确的主路径放前面"""
				# 无约束
				if name == "train" and len(PosQueryGraph) == 1:
					f1_writer.writerow([query_e, 'ns:' + PosQueryGraph[0] + ' ?x .', 1, 1])
				if one_hop_rels:
					one_hop_nocons_data = []
					logger.info("正在查找一跳无约束查询图")
					is_use_entity_cons = True
					"""查找一跳无约束"""
					for one_hop_rel in one_hop_rels:
						one_hop_rel = "ns:" + one_hop_rel + " ?x ."
						preAnswer = get_answer(topicEntityID, one_hop_rel)
						if preAnswer:
							F1 = CalculatePRF1(preAnswer, Answers)[-1]
							cos = cosin_sim(query_e, one_hop_rel)
							# 过滤掉一些查询图减少开销
							if F1 == 0 and cos < 0.3:
								continue
							# 这里还是采用SPARQL标准方式，方便后期查库和理解
							one_hop_nocons_data.append([query_e, one_hop_rel, cos, F1])
					one_hop_rels_sortbycosF1 = sorted(one_hop_nocons_data, key=lambda x: [x[2], x[3]], reverse=True)
					if one_hop_nocons_data:
						one_hop_rels_sortbyf1 = sorted(one_hop_nocons_data, key=lambda x: x[-1], reverse=True)
						f1_writer.writerows(one_hop_rels_sortbyf1)
						if one_hop_rels_sortbyf1[0][-1] != 0 and len(PosQueryGraph) > 1:
							logger.info(f"该问题是两跳问题但是一跳无约束能够找到答案")
							twohop_but_onehop += 1
						if one_hop_rels_sortbyf1[0][-1] == 1:
							logger.info(f"该问题无需加入约束即可找全答案")
							continue
					else:
						logger.info("该问题第一跳路径查不到候选答案")
						continue
					"""--------------------------------------------------------------------一跳无约束查找完毕，查询一跳带约束----------------------------------------------------------------"""
					"""处理实体约束，下一步这里可以考虑用约束选择模型来做"""
					"""在查找实体约束时可能有实体是主题实体，需要用FILTER (?x != ns:id)过滤，答案实体需要加common.topic.notable_types实体约束的问题有220个左右，"""
					"""对于实体约束common.topic.notable_types，只有加在最后的答案实体上，通过公共子串来匹配，没有可以用模型选择"""
					"""中间实体，目前发现带公共子串的那个实体更可能是实体约束，但也有例外，需要用模型选择，可以设定一个阈值（0.7）来判断是否需要加入实体约束，加入可能会导致结果不够精确"""
					"""对于不加实体约束就能查到所有答案的问题就不加实体约束,因为有些答案实体约束不一样，其次如果有性别约束尽量就不要加实体约束了，最后再处理实体约束吧"""
					# 这里暂且对top k个查询图进行后续约束加入
					one_hop_cons_data = []
					logger.info("正在查找一跳带约束查询图")
					for one_hop in one_hop_rels_sortbycosF1[:k]:  # [query_e, one_hop_rel, cos, F1]
						"""如果问的是性别 婚姻等 直接加入约束--->但需要判断约束是否真实存在于知识库中  一跳问题的约束均加在x上"""
						g_1hop_c = one_hop[1]
						"男性约束"
						if set(gender_male) & set(query_e.split()):
							# one_hop[1]格式"ns:" + one_hop_rel + " ?x ."
							g_1hop_c = one_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
							preAnswer = get_answer(topicEntityID, g_1hop_c)  # 判断是否能够查询出正确的答案
							if preAnswer:
								F1 = CalculatePRF1(preAnswer, Answers)[-1]
								cos = cosin_sim(query_e, g_1hop_c)
								one_hop_cons_data.append([query_e, g_1hop_c, cos, F1])
								if F1 == 1:
									is_use_entity_cons = False

						"女性约束"
						if set(gender_female) & set(query_e.split()):
							g_1hop_c = one_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
							preAnswer = get_answer(topicEntityID, g_1hop_c)  # 判断是否能够查询出正确的答案
							if preAnswer:
								F1 = CalculatePRF1(preAnswer, Answers)[-1]
								cos = cosin_sim(query_e, g_1hop_c)
								one_hop_cons_data.append([query_e, g_1hop_c, cos, F1])
								if F1 == 1:
									is_use_entity_cons = False

						"""实体约束"""
						if is_use_entity_cons:
							one_hop_entity_constrains = from_id_path_get_2hop_po_oName(question, topicEntityID, one_hop[1])  # [['ns:common.topic.notable_types', 'm.01m9', 'City/Town/Village'], ['ns:common.topic.notable_types', 'm.0kpys4', 'US State']]
							if one_hop_entity_constrains:
								# 这里可以对候选实体约束排个序选最高
								one_hop_entity_constrain = one_hop_entity_constrains[0]
								# [core_path,id.name] 这里是加完其他约束后根据情况再加实体约束
								g_1hop_c = g_1hop_c + f"?x ns:{one_hop_entity_constrain[0]} ns:{one_hop_entity_constrain[1]} ."
								preAnswer = get_answer(topicEntityID, g_1hop_c)  # 判断是否能够查询出正确的答案
								# print("preAnswer-------------->", preAnswer)
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_1hop_c)
									one_hop_cons_data.append([query_e, g_1hop_c, cos, F1])
						# """时间约束 主要是年份！"""
						"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
						if yearCons:  # 如果问题中有年份的时间约束
							yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
							# 至于约束的名称则需要在库里面查询 （ from ,to ）
							from_path, to_path = query_1hop_p_from_to(topicEntityID, one_hop[1],
																	  yearData)  # 选出成对的路径
							if from_path and to_path:
								g_1hop_c = g_1hop_c + 'FILTER(NOT EXISTS {?x ns:%s ?sk0} || EXISTS {?x ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
													 'FILTER(NOT EXISTS {?x ns:%s ?sk2} || EXISTS {?x ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
										   % (from_path, from_path, yearData, to_path, to_path, yearData)
								preAnswer = get_answer_new(topicEntityID, g_1hop_c)  # 判断是否能够查询出正确的答案
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_1hop_c)
									one_hop_cons_data.append([query_e, g_1hop_c, cos, F1])
									# if F1 == 1:
									# 	is_use_entity_cons = False
						# 升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）
						if order_ASC in query_e:
							# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
							paths = query_order_asc_1hop(topicEntityID, one_hop[1])
							if paths:
								# 暂时只要一个
								g_1hop_c = g_1hop_c + "?x ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (paths[0])
								preAnswer = get_answer_new(topicEntityID, g_1hop_c)  # 判断是否能够查询出正确的答案
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_1hop_c)
									one_hop_cons_data.append([query_e, g_1hop_c, cos, F1])
								# if F1 != 0:
								# 	is_use_entity_cons = False
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
								g_1hop_c = g_1hop_c + "?x ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (
									desc_path)
								preAnswer = get_answer_new(topicEntityID, g_1hop_c)  # 判断是否能够查询出正确的答案
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_1hop_c)
									one_hop_cons_data.append([query_e, g_1hop_c, cos, F1])
								# if F1 != 0:
								# 	is_use_entity_cons = False

					if one_hop_cons_data:
						one_hop_rels_cons_sortbycosF1 = sorted(one_hop_cons_data, key=lambda x: [x[2], x[3]],
															   reverse=True)
						f2_writer.writerows(one_hop_rels_cons_sortbycosF1)
						one_hop_rels_cons_sortbyf1 = sorted(one_hop_cons_data, key=lambda x: x[-1], reverse=True)
						if one_hop_rels_cons_sortbyf1[0][-1] != 0 and len(PosQueryGraph) > 1:
							logger.info(f"该问题是两跳问题但是一跳有约束能够找到答案")
							twohop_but_onehop_cons += 1
						if len(PosQueryGraph) == 1:
							logger.info("该问题是一跳问题，查找完毕")
							continue
						logger.info("一跳带约束查找完毕，接着查询第二跳无约束")
					else:
						logger.info("该问题一跳约束没有找到候选答案")
					"""--------------------------------------------------------------------一跳带约束查找完毕，查询两跳无约束----------------------------------------------------------------"""
					"""后面预测的时候根据第一跳的F1值来评估是否需要继续第二跳的查询"""
					"""获取第二跳无约束路径，从第一跳cos最大值的路径开始，后面再加约束"""
					if name == "train" and len(PosQueryGraph) == 2:
						f1_writer.writerow([query_e, 'ns:' + PosQueryGraph[0] + ' ?y .' + '?y ' + 'ns:' + PosQueryGraph[1] + ' ?x .', 1, 1])
					one_hop_in_two = one_hop_rels_sortbycosF1[0][1].replace(" ?x .", "")
					two_hop_rels = from_id_path_get_2hop_po(topicEntityID, one_hop_in_two)  # [('language.human_language.countries_spoken_in', 0.7789026498794556)]
					if two_hop_rels:
						logger.info("正在查找第二跳无约束查询图")
						two_hop_nocons_data = []
						is_use_entity_cons = True
						for two_hop_rel in two_hop_rels:
							two_hop = f"{one_hop_in_two} ?y .?y ns:{two_hop_rel} ?x ."
							preAnswer = get_answer(topicEntityID, two_hop)
							if preAnswer:
								F1 = CalculatePRF1(preAnswer, Answers)[-1]
								# 注意这里计算的是第二跳路径，没有两跳一起计算
								cos = cosin_sim(query_e, two_hop_rel)
								# 过滤掉一些查询图减少开销
								if F1 == 0 and cos < 0.3:
									continue
								two_hop_nocons_data.append([query_e, two_hop, cos, F1]) #two_hop:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						if two_hop_nocons_data:
							two_hop_rels_sortbycosF1 = sorted(two_hop_nocons_data, key=lambda x: [x[2], x[3]], reverse=True)
							two_hop_rels_sortbyF1 = sorted(two_hop_nocons_data, key=lambda x: x[-1], reverse=True)
							f1_writer.writerows(two_hop_rels_sortbyF1)
							if two_hop_rels_sortbyF1[0][1] == 1:
								logger.info(f"该问题两跳无需加入约束即可找全答案")
								continue
						else:
							logger.info("该问题第二跳路径查不到候选答案")
							continue
						"""--------------------------------------------------------------------两跳无约束查找完毕，查询两跳带约束----------------------------------------------------------------"""
						"""同样最后处理实体约束, 同样选前k个"""
						two_hop_cons_data = []
						logger.info("正在查找第二跳带约束查询图")
						# two_hop[1]:"ns:language.human_language.countries_spoken_in ?y .?y ns:language.human_language.countries_spoken_in ?x ."
						for two_hop in two_hop_rels_sortbycosF1[:k]:  # [query_e, two_hop_rel, cos, F1]

							g_2hop_c = two_hop[1]

							"""男性约束"""
							if set(gender_male) & set(query_e.split()):
								g_2hop_c = two_hop[1] + f"?x ns:people.person.gender ns:m.05zppz ."  # 男性
								preAnswer = get_answer(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_2hop_c)
									two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
									if F1 == 1:
										is_use_entity_cons = False

							"""女性约束"""
							if set(gender_female) & set(query_e.split()):
								g_2hop_c = two_hop[1] + f"?x ns:people.person.gender ns:m.02zsn ."  # 男性
								preAnswer = get_answer(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_2hop_c)
									two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
									if F1 == 1:
										is_use_entity_cons = False

							"是否结婚的约束"
							if set(marry) & set(query_e.split()):
								g_2hop_c = two_hop[1] + f"?y ns:people.marriage.type_of_union ns:m.04ztj ."  # 婚姻 约束在y上 是否都有时间限制
								preAnswer = get_answer(topicEntityID, g_2hop_c)
								if preAnswer:
									F1 = CalculatePRF1(preAnswer, Answers)[-1]
									cos = cosin_sim(query_e, g_2hop_c)
									two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
									if F1 == 1:
										is_use_entity_cons = False
							"""实体约束"""
							if is_use_entity_cons:
								"""处理约束 实体约束 约束加载第二跳的实体上"""  # 实体存在问题中 查询出第二跳的实体名称及路径
								two_hop_entity_constrains_x = from_id_path_get_3hop_po_oName_x(question, topicEntityID, two_hop[1])
								"两跳实体约束（加在x上的）"
								if two_hop_entity_constrains_x:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
									for two_hop_entity_constrain in two_hop_entity_constrains_x:
										# two_hop_entity_constrain:["ns:common.topic.notable_types", id, name]
										g_2hop_c = g_2hop_c + f"?x ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
										preAnswer = get_answer(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
										if preAnswer:
											F1 = CalculatePRF1(preAnswer, Answers)[-1]
											cos = cosin_sim(query_e, g_2hop_c)
											two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
											if F1 == 1:
												logger.info("该问题约束加在x上找全答案")
								# 在y上的实体约束
								"两跳实体约束（加在y上的）"
								two_hop_entity_constrains_y = from_id_path_get_3hop_po_oName_y(question, topicEntityID, two_hop[1])
								if two_hop_entity_constrains_y:  # 如果存在实体约束 变放弃无约束的路径 约束加在y上的
									for two_hop_entity_constrain in two_hop_entity_constrains_y:
										# two_hop_entity_constrain:[path, id, name]
										g_2hop_c = g_2hop_c + f"?y ns:{two_hop_entity_constrain[0]} ns:{two_hop_entity_constrain[1]} ."
										preAnswer = get_answer(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
										if preAnswer:
											F1 = CalculatePRF1(preAnswer, Answers)[-1]
											cos = cosin_sim(query_e, g_2hop_c)
											two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
											if F1 == 1:
												logger.info("该问题约束加在y上找全答案")
							"""主要包括两种类型的时间约束，第一种是.from和.to（只有两跳问题才有），还有一种是.date（一跳或两跳都有）"""
							if yearCons:  # 如果问题中有年份的时间约束
								yearData = yearCons[0]  # 以防万一有多个时间 但该数据集中只有一个  # 在x加上时间约束
								# 至于约束的名称则需要在库里面查询 （ from ,to ）
								from_path, to_path = query_2hop_p_from_to(topicEntityID, two_hop[1], yearData)  # 选出成对的路径
								if from_path and to_path:
									# 数据集中gender之后没有year约束但是有DESC约束
									g_2hop_c = g_2hop_c + 'FILTER(NOT EXISTS {?y ns:%s ?sk0} || EXISTS {?y ns:%s  ?sk1 . FILTER(xsd:datetime(?sk1) <= "%s-12-31"^^xsd:dateTime) })' \
														'FILTER(NOT EXISTS {?y ns:%s ?sk2} || EXISTS {?y ns:%s  ?sk3 .  FILTER(xsd:datetime(?sk3) >= "%s-01-01"^^xsd:dateTime) })}' \
											   % (from_path, from_path, yearData, to_path, to_path, yearData)
									preAnswer = get_answer_new(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
									if preAnswer:
										F1 = CalculatePRF1(preAnswer, Answers)[-1]
										cos = cosin_sim(query_e, g_2hop_c)
										two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
										# if F1 == 1:
										# 	is_use_entity_cons = False
							"""升序约束（问题中包含"first"关键字，主要是路径最后一个字段包含.from或_date,一条路径较少，主要在两条路径，不太好判断加在中间实体还是答案实体上）"""
							if order_ASC in query_e:
								# 查找包含date或from 的路径  sk0是时间 “1979”^^<http://www.w3.org/2001/XMLSchema#gYear>
								paths = query_order_asc_2hop(topicEntityID, two_hop[1])
								if paths:
									# 暂时只要一个
									g_2hop_c = g_2hop_c + "?y ns:%s ?sk0 .}ORDER BY xsd:datetime(?sk0)LIMIT 1" % (paths[0])
									preAnswer = get_answer_new(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
									if preAnswer:
										F1 = CalculatePRF1(preAnswer, Answers)[-1]
										cos = cosin_sim(query_e, g_2hop_c)
										two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
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
									g_2hop_c = g_2hop_c + "?y ns:%s ?sk0 .}ORDER BY DESC(xsd:datetime(?sk0))LIMIT 1" % (desc_path)
									preAnswer = get_answer_new(topicEntityID, g_2hop_c)  # 判断是否能够查询出正确的答案
									if preAnswer:
										F1 = CalculatePRF1(preAnswer, Answers)[-1]
										cos = cosin_sim(query_e, g_2hop_c)
										two_hop_cons_data.append([query_e, g_2hop_c, cos, F1])
										# if F1 != 0:
										# 	is_use_entity_cons = False
						if two_hop_cons_data:
							two_hop_rels_cons_sortbycosF1 = sorted(two_hop_cons_data, key=lambda x: [x[2], x[3]], reverse=True)
							f2_writer.writerows(two_hop_rels_cons_sortbycosF1)
							two_hop_rels_cons_sortbyF1 = sorted(two_hop_cons_data, key=lambda x: x[-1], reverse=True)
							if two_hop_rels_cons_sortbyF1[0][1] == 1:
								logger.info(f"该问题两跳无需加入约束即可找全答案")
								continue
						else:
							logger.info("该问题第二跳路约束径查不到候选答案")
							continue



		logger.info(f"总计：没有答案的问题{no_answer}个")
		logger.info(f"总计：是两跳问题但是一跳无约束能够找到答案的问题{twohop_but_onehop}个")
		logger.info(f"总计：是两跳问题但是一跳有约束能够找到答案的问题{twohop_but_onehop_cons}个")
		f1.close()
		f2.close()


if __name__ == '__main__':
	# for i in [1,2,3,4,5]:
	#     create("test", True, i, 0.2)
	args = parse_args_train()
	logger.add(join(args.output_path, "generate_candidate_graph_train.log"))
	create("test", 10)

	# print(CalculatePRF1([], [])[-1])
# s1 = "what does jamaican people speak"
	# s2 = ["location.country.languages_spoken",
	# 	  "location.country.administrative_divisions", "location.country.calling_code",
	# 	  "location.country.fifa_code", "location.country.official_language",
	# 	  "book.book_subject.works"]
	# print(Cal_q_path_score(s1, s2))
