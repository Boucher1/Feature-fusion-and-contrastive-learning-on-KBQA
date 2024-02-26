import argparse

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import TrainDataset, TestDataset
from model import simcse_unsup_loss, simcse_sup_loss, SimcseModel_T5_CNN, Matching_network, Matching_network_new
from transformers import BertTokenizer, AutoTokenizer, T5Tokenizer, T5EncoderModel, BertModel
import os
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import pandas as pd
from torchsummary import summary
import time

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
	parser.add_argument("--output_path", type=str, default='output')
	parser.add_argument("--lr", type=float, default=3e-5)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch_size_train", type=int, default=32)
	parser.add_argument("--batch_size_eval", type=int, default=64)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
	parser.add_argument("--max_len", type=int, default=32, help="max length of input")
	parser.add_argument("--seed", type=int, default=3407, help="random seed")
	# parser.add_argument("--train_file", type=str, default="data/webQSP/mask_data/train_allpath_mask_e.csv")
	# parser.add_argument("--dev_file", type=str, default="data/webQSP/mask_data/valid_allpath_mask_e_oversampleshuffle.csv")
	# parser.add_argument("--test_file", type=str, default="data/webQSP/mask_data/test_allpath_mask_e.csv")
	parser.add_argument("--train_file", type=str, default="data/webQSP/candid_cons/train_cons_path_shuffle.csv")
	parser.add_argument("--dev_file", type=str, default="data/webQSP/candid_cons/dev_cons_path_shuffle.csv")
	parser.add_argument("--test_file", type=str, default="data/webQSP/candid_cons/test_cons_path_shuffle.csv")
	parser.add_argument("--dataset", type=str, default="webQSP")
	parser.add_argument("--pretrain_model_path", type=str, default="pretrain_model/jina")
	parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
						default='cls', help='pooler to use')
	parser.add_argument("--train_mode", type=str, default='cons_select', choices=['unsupervise', 'supervise'],
						help="unsupervise or supervise")
	parser.add_argument("--overwrite_cache", action='store_true', default=True, help="overwrite cache")
	parser.add_argument("--do_train", action='store_true', default=True)
	parser.add_argument("--do_predict", action='store_true', default=True)


	args = parser.parse_args()
	seed_everything(args.seed)
	args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
	args.output_path = join(args.output_path, args.train_mode,
							'{}-lstm-cons_path-{}-bsz-{}-lr-{}-drop-{}'.format(args.pretrain_model_path.split('/')[-1], args.dataset, args.batch_size_train, args.lr, args.dropout))
	return args

def seed_everything(seed=3407):
	'''
	设置整个开发环境的seed
	:param seed:
	:param device:
	:return:
	'''
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# some cudnn methods can be random even after fixing the seed
	# unless you tell it to be deterministic
	torch.backends.cudnn.deterministic = True


def train(model, train_loader, dev_loader, test_loader, optimizer, args):
	logger.info("start training")
	model.train()
	device = args.device
	best = 0
	min_loss = 999
	for epoch in range(args.epochs):
		print(f"开始训练模型第{epoch}轮")
		model.train()
		for batch_idx, data in enumerate(tqdm(train_loader)):
			step = epoch * len(train_loader) + batch_idx
			# [64, 3, 128] 3应该代表的[sent0, sent0, hardneg]
			# print(data['input_ids'].size())
			# [batch, n, seq_len] -> [batch * n, sql_len]
			sql_len = data['input_ids'].shape[-1]
			# view调整张量形状，这里最后一个维度设置sql_len，前面自动设置
			input_ids = data['input_ids'].view(-1, sql_len).to(device)
			attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
			if args.pretrain_model_path.split("/")[-1] in ["roberta", "T5-base", "Sentence-T5-large", "Sentence-T5-base", "jina"]:
				token_type_ids = None
			else:
				token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)
			# [64, 768]
			out = model(input_ids, attention_mask, token_type_ids)
			if args.train_mode == 'unsupervise':
				loss = simcse_unsup_loss(out, device)
			else:
				loss = simcse_sup_loss(out, device)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			step += 1
			# 如果当前步数 step 是 args.eval_step 的倍数，则计算模型在验证集上的性能，同时记录损失和相关系数并将其写入 TensorBoard。如果当前模型表现比历史最佳结果更好，则保存模型参数到指定路径。
			if step % args.eval_step == 0:
				corrcoef = evaluate(model, dev_loader, device, args)
				logger.info('loss:{}, spearman corrcoef: {} in step {} epoch {}'.format(loss, corrcoef, step, epoch))
				writer.add_scalar('loss', loss, step)
				writer.add_scalar('corrcoef', corrcoef, step)
				# 切换到训练模式
				model.train()
				if loss < min_loss:
					# best = corrcoef
					min_loss = min_loss
					torch.save(model.state_dict(), join(args.output_path, 'simcse.pt'))
					# torch.save(model, join(args.output_path, 'simcse.pt'))

					# logger.info('higher corrcoef: {} in step {} epoch {}, save model'.format(best, step, epoch))
					logger.info('lower loss: {} in step {} epoch {}, save model'.format(min_loss, step, epoch))
		# 每个epoch测试一下
		corrcoef = evaluate(model, test_loader, device, args)
		logger.info('testset corrcoef spearman corrcoef: {} in epoch {}'.format(corrcoef, epoch))


def evaluate(model, dataloader, device, args):
	model.eval()
	sim_tensor = torch.tensor([], device=device)
	label_array = np.array([])
	with torch.no_grad():
		print("正在对验证或测试数据进行检验")
		if args.pretrain_model_path.split("/")[-1] in ["roberta", "T5-base", "Sentence-T5-large", "Sentence-T5-base", "jina"]:
			for source, target, label in tqdm(dataloader):
				# source        [batch, 1, seq_len] -> [batch, seq_len]
				source_input_ids = source.get('input_ids').squeeze(1).to(device)
				source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
				source_pred = model(source_input_ids, source_attention_mask, None)
				# target        [batch, 1, seq_len] -> [batch, seq_len]
				target_input_ids = target.get('input_ids').squeeze(1).to(device)
				target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
				target_pred = model(target_input_ids, target_attention_mask, None)
				# concat
				sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
				sim_tensor = torch.cat((sim_tensor, sim), dim=0)
				label_array = np.append(label_array, np.array(label))
		else:
			for source, target, label in tqdm(dataloader):
				# source        [batch, 1, seq_len] -> [batch, seq_len]
				source_input_ids = source.get('input_ids').squeeze(1).to(device)
				source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
				source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
				source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
				# target        [batch, 1, seq_len] -> [batch, seq_len]
				target_input_ids = target.get('input_ids').squeeze(1).to(device)
				target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
				target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
				target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
				# concat
				sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
				sim_tensor = torch.cat((sim_tensor, sim), dim=0)
				label_array = np.append(label_array, np.array(label))
	# corrcoef
	return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation

# 计算两个文本的相似度

def load_train_data_unsupervised(tokenizer, args):
	"""
	获取无监督训练语料
	"""
	logger.info('loading unsupervised train data')
	output_path = os.path.dirname(args.output_path)
	train_file_cache = join(output_path, 'train-unsupervise.pkl')
	if os.path.exists(train_file_cache) and not args.overwrite_cache:
		with open(train_file_cache, 'rb') as f:
			feature_list = pickle.load(f)
			logger.info("len of train data:{}".format(len(feature_list)))
			return feature_list
	feature_list = []
	with open(args.train_file, 'r', encoding='utf8') as f:
		lines = f.readlines()
		# lines = lines[:100]
		logger.info("len of train data:{}".format(len(lines)))
		for line in tqdm(lines):
			line = line.strip()
			feature = tokenizer([line, line], max_length=args.max_len, truncation=True, padding='max_length',
								return_tensors='pt')
			feature_list.append(feature)
	with open(train_file_cache, 'wb') as f:
		pickle.dump(feature_list, f)
	return feature_list

# 返回的是训练集编码后的[sent0, sent1. hardneg]列表
def load_train_data_supervised(tokenizer, args):
	"""
	获取NLI监督训练语料
	"""
	logger.info('loading supervised train data')
	output_path = os.path.dirname(args.output_path)
	train_file_cache = join(output_path, 'train-supervised.pkl')
	if os.path.exists(train_file_cache) and not args.overwrite_cache:
		with open(train_file_cache, 'rb') as f:
			feature_list = pickle.load(f)
			logger.info("len of train data:{}".format(len(feature_list)))
			return feature_list
	feature_list = []
	df = pd.read_csv(args.train_file, sep='\t')
	logger.info("len of train data:{}".format(len(df)))
	rows = df.to_dict('records')
	# rows = rows[:10000]
	for row in tqdm(rows):
		sent0 = row['sent0']
		sent1 = row['sent1']
		hard_neg = row['hard_neg']
		# 这里的三个句子分开进行编码了，没有一起
		feature = tokenizer([sent0, sent1, hard_neg], max_length=args.max_len, truncation=True, padding='max_length',
							return_tensors='pt')
		feature_list.append(feature)
	with open(train_file_cache, 'wb') as f:

		pickle.dump(feature_list, f)
	return feature_list

# 返回的是验证集或测试集问题和路径编码后与的得分组合的元组列表
def load_eval_data(tokenizer, args, mode):
	"""
	加载验证集或者测试集
	"""
	assert mode in ['dev', 'test'], 'mode should in ["dev", "test"]'
	logger.info('loading {} data'.format(mode))
	output_path = os.path.dirname(args.output_path)
	eval_file_cache = join(output_path, '{}.pkl'.format(mode))
	if os.path.exists(eval_file_cache) and not args.overwrite_cache:
		with open(eval_file_cache, 'rb') as f:
			feature_list = pickle.load(f)
			logger.info("len of {} data:{}".format(mode, len(feature_list)))
			return feature_list

	if mode == 'dev':
		eval_file = args.dev_file
	else:
		eval_file = args.test_file
	feature_list = []
	with open(eval_file, 'r', encoding='utf8') as f:
		lines = f.readlines()
		logger.info("len of {} data:{}".format(mode, len(lines)))
		for line in tqdm(lines):
			line = line.strip().split("\t")
			assert len(line) == 3 or len(line) == 9
			score = float(line[2])
			data1 = tokenizer(line[0].strip(), max_length=args.max_len, truncation=True, padding='max_length',
							  return_tensors='pt')
			data2 = tokenizer(line[1].strip(), max_length=args.max_len, truncation=True, padding='max_length',
							  return_tensors='pt')

			feature_list.append((data1, data2, score))
	with open(eval_file_cache, 'wb') as f:
		pickle.dump(feature_list, f)
	return feature_list


def test(args):
	tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
	assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"], \
		'pooler should in ["cls", "pooler", "last-avg", "first-last-avg"]'
	model = Matching_network_new(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(
		args.device)
	test_data = load_eval_data(tokenizer, args, 'test')
	test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
								 num_workers=args.num_workers)
	# 加载模型参数权重权重，这里model.py中加载的是模型基本的参数，配置信息，要想预测需要使用我们训练好的模型权重
	model.load_state_dict(torch.load(join(args.output_path, 'simcse.pt')))
	model.eval()
	corrcoef = evaluate(model, test_dataloader, args.device, args)
	print('testset corrcoef:{}'.format(corrcoef))


def main(args):
	# 加载模型
	# tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
	tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
	assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"], \
		'pooler should in ["cls", "pooler", "last-avg", "first-last-avg"]'
	# model = SimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(args.device)
	model = Matching_network_new(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(args.device)

	if args.do_train:
		# 加载数据集
		assert args.train_mode in ['supervise', 'unsupervise', 'cons_select'], \
			"train_mode should in ['supervise', 'unsupervise']"
		if args.train_mode in ['supervise', 'cons_select']:
			train_data = load_train_data_supervised(tokenizer, args)
		elif args.train_mode == 'unsupervise':
			train_data = load_train_data_unsupervised(tokenizer, args)
		train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
		train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
									  num_workers=args.num_workers)
		dev_data = load_eval_data(tokenizer, args, 'dev')
		dev_dataset = TestDataset(dev_data, tokenizer, max_len=args.max_len)
		dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=True,
									num_workers=args.num_workers)
		test_data = load_eval_data(tokenizer, args, 'test')
		test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
		test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
									 num_workers=args.num_workers)
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
		train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, args)
	# 测试集
	if args.do_predict:
		test_data = load_eval_data(tokenizer, args, 'test')
		test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
		test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
									 num_workers=args.num_workers)
		# 加载模型参数权重权重，这里model.py中加载的是模型基本的参数，配置信息，要想预测需要使用我们训练好的模型权重
		model.load_state_dict(torch.load(join(args.output_path, 'simcse.pt')))
		model.eval()
		corrcoef = evaluate(model, test_dataloader, args.device, args)
		logger.info('testset corrcoef:{}'.format(corrcoef))


if __name__ == '__main__':

	args = parse_args()
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)
	cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
	logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
	logger.info(args)
	writer = SummaryWriter(args.output_path)
	main(args)

	# test(args)

	# 查看模型结构
	# args = parse_args()
	# model = SimcseModel_T5(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(args.device)
	# print(model)


	# 查看特征的形状
	# tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
	# sent0 = "what is the name of <e> brother"
	# sent1 = "people.person.sibling_s"
	# hard_neg = "award.award_nominee.award_nominations"
	# # 这里的三个句子分开进行编码了，没有一起
	# feature = tokenizer([sent0, sent1, hard_neg], max_length=args.max_len, truncation=True, padding='max_length',
	# 					return_tensors='pt')
	# print(f"feature的size{feature.input_ids.size()}")
	# print(f"feature的size{feature.input_ids}")

