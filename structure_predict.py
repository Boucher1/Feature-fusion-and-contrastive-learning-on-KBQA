#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 17:05
# @Author  : Xavier Byrant
# @FileName: structure_predict.py
# @Software: PyCharm

from importlib import import_module

import torch

from transformers import AutoTokenizer


def predict_each_question(text):
    dataset = 'structure_classification/hop_classification'
    model_name = "bert_CNN"  # bert
    x = import_module('structure_classification.models.' + model_name)
    config = x.Config(dataset)


    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    text = tokenizer(text.strip(), max_length=32, truncation=True, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        text_input_ids = text.get('input_ids').squeeze(1).to(config.device)
        text_attention_mask = text.get('attention_mask').squeeze(1).to(config.device)
        text_token_type_ids = text.get('token_type_ids').squeeze(1).to(config.device)
        text_tensor = (text_input_ids, text_token_type_ids, text_attention_mask)
        out = model(text_tensor)
        predic = torch.max(out.data, 1)[1].cpu().numpy()

        # print(out.data)
        # # <class 'numpy.int64'>
        # print(predic[0])
        return predic[0]
if __name__ == '__main__':
    predict_each_question("what did dmitri mendeleev discover in 1869")
