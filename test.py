# import json
# def get_label_num(tags_file):
#     labels = []
#     with open(tags_file,'r',encoding='utf-8') as f:
#         for label in f.readlines():
#             labels.append(label.strip())
#     return labels,len(labels) #返回标签列表，标签个数
# labels,num = get_label_num('./data/tags.txt')
#
#
# def _read_tsv(cls, input_file, quotechar=None):
#     """Reads a tab separated value file."""
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = []
#         line = f.readline()
#         count = 0
#         while line:
#             if count == 5:
#                 break
#             count += 1
#             line = json.loads(line)
#             for text in line:
#                 temp = []
#                 labels = str(','.join(text['labels']))
#                 sentence = str(text['sentence'])
#                 # if labels == '':
#                 #   labels = 'DV0'
#                 temp.append(labels)
#                 temp.append(sentence)
#                 lines.append(temp)
#             line = f.readline()
#     return lines
#
# lines = _read_tsv('./data/input.json',"./data/input.json")
# print(labels,num)
# print(lines)
# for (i, line) in enumerate(lines):
#     print(i,line)
# import random
# a = [1,2,3,4,5,6,7,8,9]
# random.shuffle(a)
# print(a)

# import time
# start = time.clock()
# # 当中是你的程序
# end = time.clock()
# elapsed = (time.clock() - start)
# print("Time used:", elapsed)

# data = './output/result.txt'
# lines = open(data,'r',encoding='utf-8')
# def count_max_f(file):
#     lines = open(file,'r',encoding='utf-8').readlines()
#     max_f = 0
#     line_count = 0
#     max_line_count = 0
#     k = -12
#
#     for line in lines:
#         # if 'macro avg' in line:
#         #     print(line)
#         #     print(line.split('\t')[0].split('    ')[1].strip())
#         #     print(line.split('\t')[0].split('    ')[2].strip())
#         #     print(line.split('\t')[0].split('    ')[3].strip())
#         line_count += 1
#         if 'macro avg' in line:
#             f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(line.split('\t')[0].split('    ')[2].strip()) / (float(line.split('\t')[0].split('    ')[1].strip()) + float(line.split('\t')[0].split('    ')[2].strip()))
#             if(f1 > max_f):
#                 max_f = f1
#                 max_line_count = line_count
#
#     for i in range(13):
#         print(lines[max_line_count + k])
#         k += 1
#
#     print('The true max f1:',max_f)
#
# count_max_f(data)

# text = '啊的卡上科技大厦'
# print(text[10:20])
#
# file_road = './data/__Result_nolap.txt'
# lines = open(file_road,'r',encoding='utf-8').readlines()
# negative = 0
# positive = 0
# all = 0
# for line in lines:
#     if line.split(' ')[3].strip() == 'Negative':
#         negative += 1
#     else:
#         positive += 1
#     all += 1
#
# print(negative,positive,all,negative + positive)

# data = './data/8_features_Eng_data/8_features_Eng_All.txt'
# lines = open(data,'r',encoding='utf-8').readlines()
# type = set()
# for line in lines:
#     type.add(line.split('\\')[6])
#     type.add(line.split('\\')[7])
# print(type)
# import nltk
# # nltk.download('punkt')
# # nltk.download('averaged_perceptron_tagger')
# sen = 'you'
# print(nltk.pos_tag(nltk.word_tokenize(sen)))

# train_file = './data/8_features_Eng_data/8_features_Eng_train.txt'
# dev_file = './data/8_features_Eng_data/8_features_Eng_dev.txt'
# test_file = './data/8_features_Eng_data/8_features_Eng_test.txt'
#
# train_lines = open(train_file,'r',encoding='utf-8').readlines()
# dev_lines = open(dev_file,'r',encoding='utf-8').readlines()
# test_lines = open(test_file,'r',encoding='utf-8').readlines()
#
# def length(lines):
#     l = dict()
#     for line in lines:
#         sentence = line.split('\\')[-2]
#         if len(sentence) not in l:
#             l[len(sentence)] = 1
#         else:
#             l[len(sentence)] += 1
#     print(sorted(l.items(),key = lambda x:x[0],reverse=True))
#
# length(train_lines)
# length(dev_lines)
# length(test_lines)
# import scipy.sparse as sp
# import numpy as np
# adj = np.zeros([5, 5], dtype=float)
# adj = sp.coo_matrix(adj)
# print(adj)
# print(666)
# left_index, mid_index, right_index,entity1_index,entity2_index = [],[],[],[],[]
# for multi_index in range(1,31):
#     left_index.append(str(multi_index))
# for multi_index in range(31,61):
#     mid_index.append(str(multi_index))
# for multi_index in range(61,91):
#     right_index.append(str(multi_index))
# print(left_index)
# import tensorflow as tf
# import numpy as np
# # weights = tf.Variable(tf.random.truncated_normal([300, 300], -0.35, 0.35),name='weights_', dtype=tf.float32)
# # a = np.array(tf.random.truncated_normal([300, 300], -0.35, 0.35))
# # print(weights)
#
# node_num = 13 + 100 + 30 * 3 + 4 * 2
# adj = np.zeros([node_num, node_num], dtype=float)
#
# """
# # entity_Position
# # entity_1
# # entity_2
# # RightPos_Entity1
# # LeftPos_Entity1
# # entity1_type
# # entity1_subtype
# # entity1_head
# # RightPos_Entity2
# # LeftPos_Entity2
# # entity2_type
# # entity2_subtype
# # entity2_head
# # mark1
# # mark2
# # mark3
# # mark4
# """
# adj[1, 2], adj[2, 1] = 1, 1
# adj[1, 3], adj[3, 1] = 1, 1
# adj[1, 4], adj[4, 1] = 1, 1
# adj[1, 5], adj[5, 1] = 1, 1
# adj[1, 6], adj[6, 1] = 1, 1
# adj[1, 7], adj[7, 1] = 1, 1
# adj[2, 8], adj[8, 2] = 1, 1
# adj[2, 9], adj[9, 2] = 1, 1
# adj[2, 10], adj[10, 2] = 1, 1
# adj[2, 11], adj[11, 2] = 1, 1
# adj[2, 12], adj[12, 2] = 1, 1
# adj[3, 8], adj[8, 3] = 1, 1
# adj[4, 9], adj[9, 4] = 1, 1
# adj[5, 10], adj[10, 5] = 1, 1
# adj[6, 11], adj[11, 6] = 1, 1
# adj[7, 12], adj[12, 7] = 1, 1
# adj[13, 14], adj[14, 13] = 1, 1
# adj[15, 16], adj[16, 15] = 1, 1
#
# adj_constant = tf.constant(adj)
# adj = tf.Variable(adj, dtype=tf.float32, name='adj', trainable=False) # “constant创建的是常数，tf. Variable创建的是变量。
# # 变量属于可训练参数，在训练过程中其值会持续变化，也可以人工重新赋值，而常数的值自创建起就无法改变。”
# print(adj)
# print(adj_constant)
# import numpy as np
# node_num = 10
# adj = np.random.rand(node_num, node_num)
# print(adj)
# dj = np.zeros([node_num, node_num], dtype=float)
# print(dj)
# input_file_train = './data/ACE_ENG/train.txt'
# input_file_test = './data/ACE_ENG/test.txt'
# input_file_val = './data/ACE_ENG/dev.txt'
# mark_set = set()
# save_file = './data/ACE_ENG/marks.txt'
# lines = open(input_file_train, 'r', encoding='utf-8').readlines()
# entity_types = ['VEH', 'WEA', 'GPE', 'PER', 'LOC', 'ORG', 'FAC']
# for line in lines:
#     element_list = line.split('\\')
#     entity_position = element_list[0].strip()  # 实体相对位置
#
#     entity1_type = ''.join(element_list[6].strip())
#     entity2_type = ''.join(element_list[7].strip())
#
#     if entity2_type not in entity_types:
#         entity2_type = ''.join(line.split('\\')[8].strip())
#         entity_1_subtype = ''.join(line.split('\\')[7].strip())
#         entity_2_subtype = ''.join(line.split('\\')[9].strip())
#     mark_1L = '<' + entity1_type + '_1>'
#     mark_1R = '</' + entity1_type + '_1>'
#     mark_2L = '<' + entity2_type + '_2>'
#     mark_2R = '</' + entity2_type + '_2>'
#     mark_set.add(mark_1L)
#     mark_set.add(mark_1R)
#     mark_set.add(mark_2L)
#     mark_set.add(mark_2R)
# lines = open(input_file_test, 'r', encoding='utf-8').readlines()
# for line in lines:
#     element_list = line.split('\\')
#     entity_position = element_list[0].strip()  # 实体相对位置
#
#     entity1_type = ''.join(element_list[6].strip())
#     entity2_type = ''.join(element_list[7].strip())
#
#     if entity2_type not in entity_types:
#         entity2_type = ''.join(line.split('\\')[8].strip())
#         entity_1_subtype = ''.join(line.split('\\')[7].strip())
#         entity_2_subtype = ''.join(line.split('\\')[9].strip())
#     mark_1L = '<' + entity1_type + '_1>'
#     mark_1R = '</' + entity1_type + '_1>'
#     mark_2L = '<' + entity2_type + '_2>'
#     mark_2R = '</' + entity2_type + '_2>'
#     mark_set.add(mark_1L)
#     mark_set.add(mark_1R)
#     mark_set.add(mark_2L)
#     mark_set.add(mark_2R)
# lines = open(input_file_val, 'r', encoding='utf-8').readlines()
# for line in lines:
#     element_list = line.split('\\')
#     entity_position = element_list[0].strip()  # 实体相对位置
#
#     entity1_type = ''.join(element_list[6].strip())
#     entity2_type = ''.join(element_list[7].strip())
#
#     if entity2_type not in entity_types:
#         entity2_type = ''.join(line.split('\\')[8].strip())
#         entity_1_subtype = ''.join(line.split('\\')[7].strip())
#         entity_2_subtype = ''.join(line.split('\\')[9].strip())
#     mark_1L = '<' + entity1_type + '_1>'
#     mark_1R = '</' + entity1_type + '_1>'
#     mark_2L = '<' + entity2_type + '_2>'
#     mark_2R = '</' + entity2_type + '_2>'
#     mark_set.add(mark_1L)
#     mark_set.add(mark_1R)
#     mark_set.add(mark_2L)
#     mark_set.add(mark_2R)
#
# print(len(mark_set),mark_set)
# save = open(save_file,'w',encoding='utf-8')
# for mark in mark_set:
#     save.write(mark + '\n')

# data_file = './output/PRF-bert-80.12.txt'
# lines = open(data_file,'r',encoding='utf-8').readlines()
# for line in lines:
#     print(line)
#     if 'macro avg' in line:
#         element_list = line.split('    ')
#         p = float(element_list[1])
#         r = float(element_list[2])
#         f1 = (2 * p * r) / ( p + r )
        # f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(
        #     line.split('\t')[0].split('    ')[2].strip()) / (
        #              float(line.split('\t')[0].split('    ')[1].strip()) + float(
        #          line.split('\t')[0].split('    ')[2].strip()))

# import numpy as np
# adj_word = np.random.rand(100, 100)
# print(adj_word)
# adj = []
# for i in range(100):
#     x = np.random.normal(loc=1, scale=0.5, size=100)
#     adj.append(np.array(x))
# adj = np.array(adj)
# print(adj)

# import datetime
# import os
# ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
# # current_time = datetime.datetime.now().strftime(ISOTIMEFORMAT)
# print('Waiting……')
# while True:
#     current_time = datetime.datetime.now().strftime(ISOTIMEFORMAT)
#     if str(current_time) == '2022-05-04 03:00:00':
#         os.system('python Excute_main_predict_streamlined.py')
#         break

import tensorflow as tf

weights = tf.Variable(tf.random.truncated_normal([24, 24], -1, 1),name='weights_', dtype=tf.float32)
print(222)
