# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from sklearn import metrics
import collections
import csv
import datetime
import os
import bert_master.modeling as modeling
import bert_master.optimization as optimization
import bert_master.tokenization as tokenization
import tensorflow as tf
import pandas as pd
import jieba
import jieba.posseg
import platform
import numpy as np
import pickle
import json
import nltk
import random
from gcn_layers import *
from gcn_utils import *
from keras_applications import *

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

#allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片
# per_process_gpu_memory_fraction
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
#设置每个GPU应该拿出多少容量给进程使用，0.5代表 50%


if os.name == 'nt':
    bert_path = './chinese_L-12_H-768_A-12/'
    root_path = './'
else:
    bert_path = './chinese_L-12_H-768_A-12/'
    root_path = './'

# %%
## Required parameters
flags.DEFINE_string(
    "data_dir", "./data/ACE_CHN",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", bert_path + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "relation_extraction", "The name of the task to train.")

flags.DEFINE_string("vocab_file", bert_path + "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./output/models",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", bert_path + "bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 148,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2048.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 2000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 2000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

def getPOSleft(text):
    POS = []
    for words in text:
        words = words.strip()
        if words == '#' or len(words) == 0:
            POS.append('None')
        else:
            word = words[-1].strip()
            for word, flag in jieba.posseg.cut(word):
                POS.append(flag)
    return POS


def getPOSright(text):
    POS = []
    for words in text:
        words = words.strip()
        if words == '#' or len(words) == 0:
            POS.append('None')
        else:
            word = words[0].strip()
            for word, flag in jieba.posseg.cut(word):
                POS.append(flag)
    return POS

# def Eng_get_POS_left(text):
#     POS = []
#     for word in text:
#         text_list = nltk.word_tokenize(word)
#     # POS = nltk.pos_tag(text_list)[0][1]  # 打标签
#         word2POS = nltk.pos_tag(text_list)
#         P = word2POS[0][1]
#         POS.append(P)
#     return POS # 返回字符串
#
# def Eng_get_POS_right(text):
#     POS = []
#     for word in text:
#         text_list = nltk.word_tokenize(word)
#     # POS = nltk.pos_tag(text_list)[0][1]  # 打标签
#         word2POS = nltk.pos_tag(text_list)
#         P = word2POS[-1][1]
#         POS.append(P)
#     return POS # 返回字符串

# def get_POS(text):
#     text_list = nltk.word_tokenize(text)
#     # POS = nltk.pos_tag(text_list)[0][1]  # 打标签
#     POS = nltk.pos_tag(text_list)[0][1]
#     return POS # 返回字符串


def concate(data_1, data_2):
    result = []
    for i in range(len(data_1)):
        result.append(data_1[i] + '_' + data_2[i])
    return result


def getNumClasses():
    tag_path = os.path.join(FLAGS.data_dir, "tags.txt")
    f = open(tag_path, 'r', encoding='utf-8')
    lines = f.readlines()
    label = []
    for line in lines:
        label.append(line.strip())
    f.close()
    return len(label)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    # def __init__(self, guid, text_a, mark_index, single_feature_index,combine_feature_index, word_index, text_b=None, label=None):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        # self.mark_index = mark_index
        # self.single_feature_index = single_feature_index
        # self.combine_feature_index = combine_feature_index
        # self.word_index = word_index


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 mark_index,
                 single_feature_index,
                 combine_feature_index,
                 formal_12,
                 latter_12,
                 random_12,
                 odd_number,
                 even_number,
                 word_index,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.mark_index = mark_index,
        self.single_feature_index = single_feature_index,
        self.combine_feature_index = combine_feature_index,
        self.formal_12 = formal_12,
        self.latter_12 = latter_12,
        self.random_12 = random_12,
        self.odd_number = odd_number,
        self.even_number = even_number,
        self.word_index = word_index,
        self.is_real_example = is_real_example

def get_relation_entity_types(input_file):
    entity_types_set = set()
    relation_set = set()
    lines = open(input_file, 'r', encoding='utf-8').readlines()
    for line in lines:
        element_set = line.split(' ')
        relation = element_set[-1].strip()
        entity_1_type = element_set[6].strip()
        entity_2_type = element_set[7].strip()
        entity_types_set.add(entity_1_type)
        entity_types_set.add(entity_2_type)
        relation_set.add(relation)
    return relation_set, entity_types_set

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        relation_set, entity_types = get_relation_entity_types(input_file)
        data = []
        lines = open(input_file, 'r', encoding='utf-8').readlines()
        # lines = lines[0:1000]
        for line in lines:
            element_list = line.split(' ')
            entity_position = element_list[0].strip()  # 实体相对位置
            left = ''.join(element_list[1].strip())  # 左边部分
            entity_1 = ''.join(element_list[2].strip())  # 实体1
            middle = ''.join(element_list[3].strip())
            entity_2 = ''.join(element_list[4].strip())
            right = ''.join(element_list[5].strip())
            entity_1_type = ''.join(element_list[6].strip())
            entity_2_type = ''.join(element_list[7].strip())
            entity_1_subtype = ''.join(element_list[8].strip())
            entity_2_subtype = ''.join(element_list[9].strip())
            if entity_2_type not in entity_types:
                entity_2_type = ''.join(element_list[8].strip())
                entity_1_subtype = ''.join(element_list[7].strip())
                entity_2_subtype = ''.join(element_list[9].strip())
            entity_1_head = ''.join(element_list[10].strip())
            entity_2_head = ''.join(element_list[11].strip())
            sentence = ''.join(element_list[12].strip())
            relation = element_list[13].strip()

            if relation not in relation_set:
                print(line)
            # entityPosition = str(entityPosition)
            data.append([left, entity_1, middle, entity_2, right, entity_1_type, entity_2_type, entity_1_subtype,
                         entity_2_subtype, entity_1_head, entity_2_head, entity_position, sentence, relation])
            # data.append([id, sentence, relation])
        df = pd.DataFrame(data, columns=["left", "entity_1", "middle", "entity_2", "right", "entity1_Type",
                                             "entity2_Type", "entity1_Subtype",
                                             "entity2_Subtype", "entity1_Head", "entity2_Head", "entityPosition",
                                             "sentence", "relation"])
        x_text1 = left = df['left'].tolist()
        x_text2 = entity_1 = df['entity_1'].tolist()
        x_text3 = middle = df['middle'].tolist()
        x_text4 = entity_2 = df['entity_2'].tolist()
        x_text5 = right = df['right'].tolist()
        x_text6 = entity_1_type = df['entity1_Type'].tolist()
        x_text7 = entity_2_type = df['entity2_Type'].tolist()
        x_text8 = entity1_Subtype = df['entity1_Subtype'].tolist()
        x_text9 = entity2_Subtype = df['entity2_Subtype'].tolist()
        x_text10 = entity1_Head = df['entity1_Head'].tolist()
        x_text11 = entity2_Head = df['entity2_Head'].tolist()
        x_text12 = entityPosition = df['entityPosition'].tolist()
        x_text13 = sentence = df['sentence'].tolist()
        labels = df['relation']

        Entity_1 = x_text2
        Entity_2 = x_text4
        Type_Entity1 = x_text6
        Type_Entity2 = x_text7
        Subtype_Entity1 = x_text8
        Subtype_Entity2 = x_text9
        Head_Entity1 = x_text10
        Head_Entity2 = x_text11
        LeftPos_Entity1 = getPOSleft(x_text1)
        RightPos_Entity1 = getPOSright(x_text3)
        LeftPos_Entity2 = getPOSleft(x_text3)
        RightPos_Entity2 = getPOSright(x_text5)

        Featute_1 = concate(RightPos_Entity1, Type_Entity1)
        Featute_2 = concate(LeftPos_Entity1, Type_Entity1)
        Featute_3 = concate(RightPos_Entity2, Type_Entity2)
        Featute_4 = concate(LeftPos_Entity2, Type_Entity2)
        Featute_5 = concate(Type_Entity1, Type_Entity2)
        Featute_6 = concate(Subtype_Entity1, Subtype_Entity2)
        Featute_7 = concate(Head_Entity1, Head_Entity2)
        Featute_8 = x_text12

        return_sentence = x_text13

        return (entityPosition, entity_1, entity_1_type, entity1_Subtype, entity1_Head, RightPos_Entity1, LeftPos_Entity1,
                entity_2, entity_2_type, entity2_Subtype, entity2_Head, RightPos_Entity2, LeftPos_Entity2),\
               (Featute_1, Featute_2, Featute_3, Featute_4, Featute_5, Featute_6, Featute_7, Featute_8), return_sentence, labels

        # return (Featute_1, Featute_2, Featute_3, Featute_4, Featute_5, Featute_6, Featute_7, Featute_8, Entity_1,
        #         Entity_2, Type_Entity1, Type_Entity2,
        #         Subtype_Entity1, Subtype_Entity2, Head_Entity1, Head_Entity2), return_sentence, labels


class Relation_Extraction(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "train.json")), "train")
            self._read_tsv(os.path.join(data_dir,'train.txt')), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir,'dev.txt')), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir)), "test")

    def get_labels(self):
        """See base class."""
        tag_path = os.path.join(FLAGS.data_dir, "tags.txt")
        f = open(tag_path, 'r', encoding='utf-8')
        lines = f.readlines()
        label = []
        for line in lines:
            label.append(line.strip())
        f.close()
        return label

    def _create_examples(self, lines, set_type):
        """得到的lines是一个元素为字典形式的列表，sentence是句子，labels是一个含有多标签的列表"""
        """Creates examples for the training and dev sets."""
        examples = []
        count_sum = 0
        for i in range(len(lines[2])):
            count_sum += 1

            # Only the test set has a header
            # if set_type == "test" and i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            # if set_type == "test":
            #     text_a = tokenization.convert_to_unicode(line[1])
            #     label = "0"
            # else:
            #     text_a = tokenization.convert_to_unicode(line[3])
            #     label = tokenization.convert_to_unicode(line[1])

            # (entity_Position, Entity_1, Entity_2, Type_Entity1, Type_Entity2,
            #  Subtype_Entity1, Subtype_Entity2, Head_Entity1, Head_Entity2, LeftPos_Entity1, RightPos_Entity1,
            #  LeftPos_Entity2, RightPos_Entity2, left, middle, right), return_sentence, labels


            # entity_1 = lines[0][8][i]
            # entity_2 = lines[0][9][i]
            # entity1_type = lines[0][10][i]
            # entity2_type = lines[0][11][i]
            # entity1_subtype = lines[0][12][i]
            # entity2_subtype = lines[0][13][i]
            # entity1_head = lines[0][14][i]
            # entity2_head = lines[0][15][i]
            # sentence = lines[1][i]

            f1 = entityPosition = lines[0][0][i]
            f2 = entity_1 = lines[0][1][i]
            f3 = entity_1_type = lines[0][2][i]
            f4 = entity1_Subtype = lines[0][3][i]
            f5 = entity1_Head = lines[0][4][i]
            f6 = RightPos_Entity1 = lines[0][5][i]
            f7 = LeftPos_Entity1 = lines[0][6][i]
            f8 = entity_2 = lines[0][7][i]
            f9 = entity_2_type = lines[0][8][i]
            f10 = entity2_Subtype = lines[0][9][i]
            f11 = entity2_Head = lines[0][10][i]
            f12 = RightPos_Entity2 = lines[0][11][i]
            f13 = LeftPos_Entity2 = lines[0][12][i]
            F1 = RightPos_Entity1_Type_Entity1 = lines[1][0][i]
            F2 = LeftPos_Entity1_Type_Entity1 = lines[1][1][i]
            F3 = RightPos_Entity2_Type_Entity2 = lines[1][2][i]
            F4 = LeftPos_Entity2_Type_Entity2 = lines[1][3][i]
            F5 = Type_Entity1_Type_Entity2 = lines[1][4][i]
            F6 = Subtype_Entity1_Subtype_Entity2 = lines[1][5][i]
            F7 = Head_Entity1_Head_Entity2 = lines[1][6][i]
            sentence = lines[2][i]

            # mark_1L = ' <L_' + LeftPos_Entity1 + '_' + entity1_type + '_1> '
            # mark_1R = ' </R_' + RightPos_Entity1 + '_' + entity1_type + '_1> '
            # mark_2L = ' <L_' + LeftPos_Entity2 + '_' + entity2_type + '_2> '
            # mark_2R = ' </R_' + RightPos_Entity2 + '_' + entity2_type + '_2> '
            mark_1L = ' <' + entity_1_type + '_1> '
            mark_1R = ' </' + entity_1_type + '_1> '
            mark_2L = ' <' + entity_2_type + '_2> '
            mark_2R = ' </' + entity_2_type + '_2> '
            entity1_instead = mark_1L + entity_1 + mark_1R
            entity2_instead = mark_2L + entity_2 + mark_2R
            # if 'Ward Armstrong, D-Henry County' in sentence:
            #     print("测试中！")
            # sentence = clean_str(sentence)
            if sentence.startswith("I'm"):
                sentence = sentence.replace("I'm", "I am")
            try:
                if len(entity_1) < len(entity_2):
                    sentence = sentence.replace(entity_2, entity2_instead, 1)
                    if entity_1 in mark_2L or entity_1 in mark_2R:
                        entity_1 = entity_1 + " "
                    if "R-" in sentence or "I've" in sentence or "I-" in sentence or "I'd" in sentence or "I'll" in sentence or "I'm" in sentence or "R-Loudoun" in sentence or "B's" in sentence or "B's kid" in sentence:
                        sentence = sentence.replace("R-", "R -").replace("I've", "I have").replace("I-", "I -").replace(
                            "I'd", "I 'd").replace("I'll", "I 'll").replace("I'm", "I 'm").replace("R-Loudoun",
                                                                                                   "R -Loudoun").replace(
                            "B's", "B 's")  # .replace("B 's kid","B's kid")
                    if entity_1 == "B's kid":
                        entity_1 = "B 's kid"
                    sentence = sentence.replace(entity_1, entity1_instead, 1)
                elif entity_2 in entity_1 and sentence.index(entity_1) == sentence.index(entity_2):
                    sentence = sentence.replace(entity_1, entity1_instead)
                    if entity_2 in mark_1L or entity_2 in mark_1R:
                        entity_2 = entity_2 + " "
                    if "D-Fairfax" in sentence or "R-Cape" in sentence or "D-Henry" in sentence:
                        sentence = sentence.replace("D-Fairfax", "D -Fairfax").replace("R-Cape", "R -Cape").replace(
                            "D-Henry", "D -Henry")
                    sentence = sentence.replace(entity_2, entity2_instead)
                elif sentence.index(entity_1) == sentence.index(entity_2) and len(entity_1) == len(entity_2):
                    sentence = sentence.replace(entity_1, entity1_instead, 1)
                    if entity_2 in mark_1L or entity_2 in mark_1R:
                        entity_2 = entity_2 + " "
                    sentence = sentence.replace(entity_2, entity2_instead, 1)
                elif sentence.index(entity_1) != sentence.index(entity_2):
                    sentence = sentence.replace(entity_1, entity1_instead, 1)
                    if entity_2 in mark_1L or entity_2 in mark_1R:
                        entity_2 = entity_2 + " "
                    if " dump B" in sentence or "I've" in sentence or "I-" in sentence or "I'd" in sentence or "I'll" in sentence or "I'm" in sentence or "R-Loudoun" in sentence or "D-Fairfax" in sentence:
                        sentence = sentence.replace(" dump B", " dump B ").replace("I've", "I have").replace("I-",
                                                                                                             "I -").replace(
                            "I'd", "I 'd").replace("I'll", "I 'll").replace("I'm", "I 'm").replace("R-Loudoun",
                                                                                                   "R -Loudoun").replace(
                            "D-Fairfax", "D -Fairfax")
                    if "I have seen in politics" in sentence:
                        sentence = sentence.replace("I have", "I've")
                    sentence = sentence.replace(entity_2, entity2_instead, 1)
                else:
                    sentence = sentence.replace(entity_1, entity1_instead, 1)
                    if entity_2 in mark_1L or entity_2 in mark_1R:
                        entity_2 = entity_2 + " "
                    sentence = sentence[0:sentence.index(entity1_instead) + len(entity1_instead)] + sentence[
                                                                                                    sentence.index(
                                                                                                        entity1_instead) + len(
                                                                                                        entity1_instead):].replace(
                        entity_2, entity2_instead, 1)
            except:
                print(sentence, '***1:', entity_1, '***2:', entity_2)
            # all_features = entity_Position + ' ' + entity_1 + ' ' + entity_2 + ' ' + RightPos_Entity1 + ' ' + LeftPos_Entity1 + ' ' + entity1_type + ' ' + \
            #                entity1_subtype + ' ' + entity1_head + ' ' + RightPos_Entity2 + ' ' + LeftPos_Entity2 + ' ' + \
            #                entity2_type + ' ' + entity2_subtype + ' ' + entity2_head
            # all_features = entity_Position + ' ' + RightPos_Entity1 + ' ' + LeftPos_Entity1 + ' ' + entity1_type + ' ' + \
            #                entity1_subtype + ' ' + entity1_head + ' ' + RightPos_Entity2 + ' ' + LeftPos_Entity2 + ' ' + \
            #                entity2_type + ' ' + entity2_subtype + ' ' + entity2_head
            # all_features = f8 + ' ' + sentence + ' [SEP]' + ' [CLS]' + ' ' + f8 + ' ' + f1 + ' ' + f2 + ' ' + f3 + ' ' + f4 + ' ' + f5 + ' ' + f6 + ' ' + f7 + ' ' + f9 + ' ' + f10 + ' '  + f11 + ' ' + f12 + ' ' + f13 + ' ' + f14 + ' ' + sentence
            all_features = f1 + " " + f2 + " " + f3 + " " + f4 + " " + f5 + " " + f6 + " " + f7 + " " + f8 \
                           + " " + f9 + " " + f10 + " " + f11 + " " + f12 + " " + f13 \
                           + " " + F1 + " " + F2 + " " + F3 + " " + F4 + " " + F5 + " " + F6 + " " + F7 + " " +  sentence
            # 原子特征占18位 复合特征占24位 标签占4位 句子占100位
            all_features = all_features.strip()
            if count_sum <= 10:
                print('输入实例', count_sum, ":", all_features)

            # text_a = tokenization.convert_to_unicode(all_features)
            # mark_index = []
            # single_feature_index = []
            # combine_feature_index = []
            # word_index = []
            #
            # all_features_list = all_features.split(" ")
            #
            # # print("all_features_list:", all_features_list)
            #
            # try:
            #     mark_index.append(str(all_features_list.index(mark_1L) + 1))
            # except:
            #     mark_index.append(str(random.randint(42, 146)))
            # try:
            #     mark_index.append(str(all_features_list.index(mark_1R) + 1))
            # except:
            #     mark_index.append(str(random.randint(42, 146)))
            #
            # try:
            #     mark_index.append(str(all_features_list.index(mark_2L) + 1))
            # except:
            #     mark_index.append(str(random.randint(42, 146)))
            #
            # try:
            #     mark_index.append(str(all_features_list.index(mark_2L) + 1))
            # except:
            #     mark_index.append(str(random.randint(42, 146)))
            #
            # for single_feature_id in range(0, 19):
            #     single_feature_index.append(str(single_feature_id))
            # for combine_feature_id in range(19, 44):
            #     combine_feature_index.append(str(combine_feature_id))

            # 原子特征 + 加标签的句子
            text_a = tokenization.convert_to_unicode(all_features)

            # for k in range(44,146):
            #     if str(k) not in mark_index:
            #         word_index.append(str(k))
            # if len(word_index) < 100:
            #     for add_word in range(0, (100 - len(word_index))):
            #         word_index.append(str(len(all_features_list)))
            # if len(word_index) > 100:
            #     word_index = word_index[0:100]


            # mark_index = tokenization.convert_to_unicode(" ".join(mark_index))
            # single_feature_index = tokenization.convert_to_unicode(" ".join(single_feature_index))
            # combine_feature_index = tokenization.convert_to_unicode(" ".join(combine_feature_index))
            # word_index = tokenization.convert_to_unicode(" ".join(word_index))

            labels = []
            label = lines[3][i].strip()
            labels.append(tokenization.convert_to_unicode(label))

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))

            # examples.append(
            #     InputExample(guid=guid, text_a=text_a, mark_index=mark_index, single_feature_index=single_feature_index,combine_feature_index=combine_feature_index,
            #                  word_index=word_index, text_b=None, label=labels))
            # examples.append(
            #     InputExample(guid=guid, text_a=text_a, feature_index=feature_index,
            #                  # mark_index=mark_index, word_index=word_index, left=left, entity1=entity1, mid=mid, entity2=entity2, right=right, all_atomic=all_atomic, left_index=left_index, entity1_index=entity1_index, mid_index=mid_index, entity2_index=entity2_index,right_index=right_index,
            #                  text_b=None, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        # mark_index, feature_index, head_index, n_gram_index
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # mark_index = tokenizer.tokenize(example.mark_index)
    # feature_index = tokenizer.tokenize(example.feature_index)
    # mark_ids = []
    # for mark_id in example.mark_index.split(' '):
    #     mark_ids.append(str(mark_id))
    # mark_index = mark_ids

    # head_ids = []
    # for head_id in example.head_index.split(' '):
    #     head_ids.append(str(head_id))
    # head_index = head_ids
    # print("******:",example.n_gram_index)

    # word_ids = []
    # for word_id in example.word_index.split(' '):
    #     word_ids.append(str(word_id))
    # word_index = word_ids
    # print("******:", example.n_gram_index)
    # n_gram_index = tokenizer.tokenize(example.n_gram_index)


    mark_list = []
    mark_file = os.path.join(FLAGS.data_dir,'marks.txt')
    ms = open(mark_file, 'r', encoding='utf-8').readlines()
    for m in ms:
        mark_list.append(m.strip())
    tokens_c = tokenizer.tokenize(example.text_a)
    tokens_a = []
    remember_list = []
    flag = 0
    word = ''
    for each_f in tokens_c:
        if each_f == '<' or flag == 1:
            remember_list.append(each_f)
            if each_f == '>':
                flag = 0
                word += each_f
                if word not in mark_list:
                    for origin_mark in remember_list:
                        tokens_a.append(origin_mark)
                    word = ''
                    remember_list = []
                else:
                    tokens_a.append(word)
                    word = ''
            else:
                if "##" in each_f:
                    each_f = each_f.replace('##', '')
                flag = 1
                word += each_f
        else:
            tokens_a.append(each_f)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    mark_index = []
    single_feature_index = []
    combine_feature_index = []
    word_index = []
    formal_12 = []
    latter_12 = []
    random_12 = []
    odd_number = []
    even_number = []
    for id in range(0, 21):
        single_feature_index.append(str(id))

    for id in range(21, 61):
        combine_feature_index.append(str(id))

    for id_2, seg in enumerate(tokens):
        if seg in mark_list:
            mark_index.append(id_2)

    for id_word, seg in enumerate(tokens[id + 1:]):
        if id_word + id + 1 in mark_index:
            continue
        else:
            word_index.append(id_word + id + 1)

    single_feature_length = len(single_feature_index)
    if single_feature_length > 21:
        single_feature_index = single_feature_index[0:21]
    if single_feature_length < 21:
        for add_id in range(21 - single_feature_length):
            single_feature_index.append(0)

    combine_feature_length = len(combine_feature_index)
    if combine_feature_length > 40:
        combine_feature_index = combine_feature_index[0:40]
    if combine_feature_length < 40:
        for add_id in range(40 - combine_feature_length):
            combine_feature_index.append(0)

    mark_length = len(mark_index)
    if mark_length > 4:
        mark_index = mark_index[0:4]
    if mark_length < 4:
        for add_id in range(4 - mark_length):
            mark_index.append(0)

    word_length = len(word_index)
    if word_length > 100:
        word_index = word_index[0:100]
    if word_length < 100:
        for add_id in range(100 - word_length):
            word_index.append(0)
    for formal in range(1, 21):
        formal_12.append(formal)
    for formal in range(21, 41):
        latter_12.append(formal)
    for formal in range(1, 21):
        random_12.append(np.random.randint(1, 40))

    for formal in range(1, 41):
        if formal % 2 == 0:
            even_number.append(formal)
        else:
            odd_number.append(formal)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    ##label是一个列表
    # label_id = label_map[example.label]

    label_id = [0] * len(label_map)
    if len(example.label) > 0:
        for label in example.label:
            label_index = label_map[label]
            label_id[label_index] = 1

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))
        tf.logging.info("mark_index: %s" % " ".join([str(x) for x in mark_index]))
        tf.logging.info("feature_index: %s" % " ".join([str(x) for x in single_feature_index]))
        tf.logging.info("combine_feature_index: %s" % " ".join([str(x) for x in combine_feature_index]))
        tf.logging.info("formal_12: %s" % " ".join([str(x) for x in formal_12]))
        tf.logging.info("latter_12: %s" % " ".join([str(x) for x in latter_12]))
        tf.logging.info("random_12: %s" % " ".join([str(x) for x in random_12]))
        tf.logging.info("odd_number: %s" % " ".join([str(x) for x in odd_number]))
        tf.logging.info("even_number: %s" % " ".join([str(x) for x in even_number]))
        tf.logging.info("word_index: %s" % " ".join([str(x) for x in word_index]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        mark_index=mark_index,
        single_feature_index=single_feature_index,
        combine_feature_index=combine_feature_index,
        formal_12 = formal_12,
        latter_12 = latter_12,
        random_12 = random_12,
        odd_number=odd_number,
        even_number=even_number,
        word_index=word_index,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        label_ids = feature.label_id
        features["label_ids"] = create_int_feature(label_ids)

        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        feature.single_feature_index = list(feature.single_feature_index)
        feature.single_feature_index = [int(x) for x in feature.single_feature_index[0]]

        feature.combine_feature_index = list(feature.combine_feature_index)
        feature.combine_feature_index = [int(x) for x in feature.combine_feature_index[0]]

        feature.formal_12 = list(feature.formal_12)
        feature.formal_12 = [int(x) for x in feature.formal_12[0]]

        feature.latter_12 = list(feature.latter_12)
        feature.latter_12 = [int(x) for x in feature.latter_12[0]]

        feature.random_12 = list(feature.random_12)
        feature.random_12 = [int(x) for x in feature.random_12[0]]

        feature.odd_number = list(feature.odd_number)
        feature.odd_number = [int(x) for x in feature.odd_number[0]]

        feature.even_number = list(feature.even_number)
        feature.even_number = [int(x) for x in feature.even_number[0]]

        feature.mark_index = list(feature.mark_index)
        feature.mark_index = [int(x) for x in feature.mark_index[0]]

        feature.word_index = list(feature.word_index)
        feature.word_index = [int(x) for x in feature.word_index[0]]

        features["mark_index"] = create_int_feature(feature.mark_index)
        features["single_feature_index"] = create_int_feature(feature.single_feature_index)
        features["combine_feature_index"] = create_int_feature(feature.combine_feature_index)
        features["formal_12"] = create_int_feature(feature.formal_12)
        features["latter_12"] = create_int_feature(feature.latter_12)
        features["random_12"] = create_int_feature(feature.random_12)
        features["odd_number"] = create_int_feature(feature.odd_number)
        features["even_number"] = create_int_feature(feature.even_number)
        features["word_index"] = create_int_feature(feature.word_index)
        # features["left"] = create_int_feature(feature.left)
        # features["entity1"] = create_int_feature(feature.entity1)
        # features["mid"] = create_int_feature(feature.mid)
        # features["entity2"] = create_int_feature(feature.entity2)
        # features["right"] = create_int_feature(feature.right)
        # features["left_index"] = create_int_feature(feature.left)
        # features["entity1_index"] = create_int_feature(feature.entity1)
        # features["mid_index"] = create_int_feature(feature.mid)
        # features["entity2_index"] = create_int_feature(feature.entity2)
        # features["right_index"] = create_int_feature(feature.right)
        #
        # features["all_atomic"] = create_int_feature(feature.all_atomic)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close() # 把输入特征写进文件，并在下一步进行读取


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder): # 传入训练文件
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([getNumClasses()], tf.int64),  ##固定label_ids长度
        "mark_index": tf.FixedLenFeature([4], tf.int64),
        "single_feature_index": tf.FixedLenFeature([21], tf.int64),
        "combine_feature_index": tf.FixedLenFeature([40], tf.int64),
        "word_index": tf.FixedLenFeature([100], tf.int64),
        "formal_12": tf.FixedLenFeature([20], tf.int64),
        "latter_12": tf.FixedLenFeature([20], tf.int64),
        "random_12": tf.FixedLenFeature([20], tf.int64),
        "odd_number": tf.FixedLenFeature([20], tf.int64),
        "even_number": tf.FixedLenFeature([20], tf.int64),
        # "left_index": tf.FixedLenFeature([30], tf.int64),
        # "entity1_index": tf.FixedLenFeature([4], tf.int64),
        # "mid_index": tf.FixedLenFeature([30], tf.int64),
        # "entity2_index": tf.FixedLenFeature([4], tf.int64),
        # "right_index": tf.FixedLenFeature([30], tf.int64),
        # "left": tf.FixedLenFeature([30], tf.int64),
        # "entity1": tf.FixedLenFeature([4], tf.int64),
        # "mid": tf.FixedLenFeature([30], tf.int64),
        # "entity2": tf.FixedLenFeature([4], tf.int64),
        # "right": tf.FixedLenFeature([30], tf.int64),
        # "all_atomic": tf.FixedLenFeature([13], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model_original(bert_config, is_training, input_ids, input_mask, segment_ids,
                          labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    # output_layer = model.get_pooled_output()  # 主干模型获得的模型输出
    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value  # bert的输出

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # 分类层
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()  # 主干模型获得的模型输出

    hidden_size = output_layer.shape[-1].value  # bert的输出大小

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # 分类层
        logits = tf.nn.bias_add(logits, output_bias)

        """在此处改为多标签分类的sigmoid激活函数"""
        # probabilities = tf.nn.sigmoid(logits)  # 该值是输出的预测结果
        probabilities = tf.nn.softmax(logits)  # 该值是输出的预测结果
        labels = tf.cast(labels, tf.float32)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # 可以使用focal loss损失函数解决样本不均衡的问题！！！！！！！！！！！！！！
        # def focal_loss(logits, labels, gamma = 2.0):
        #     '''
        #     :param logits:  [batch_size, n_class]
        #     :param labels: [batch_size]
        #     :return: -(1-y)^r * log(y)
        #     '''
        #     softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        #     labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
        #     prob = tf.gather(softmax, labels)
        #     weight = tf.pow(tf.subtract(1., prob), gamma)
        #     loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        #     return loss

        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)

class CNN_model():
    def __init__(self, sentence, filter_sizes, num_filters, vocab_size, text_embedding_size,
                 l2_reg_lambda=0.0001):
        self.sentence = sentence
        self.dropout_keep_prob = 0.7

        # Embedding layer
        with tf.device('/gpu:1'), tf.variable_scope('text-embedding'):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, text_embedding_size], -0.25, 0.25), name='W_text')
            self.text_embedding = tf.nn.embedding_lookup(self.W_text, self.sentence, name='sentence')

    def pro(self):
        # return self.loss, self.losses, self.scores, self.probability
        return self.h_drop

class GNN_model_conv():
    def __init__(self, embedding, embedding_mark, embedding_single_feature, embedding_combine_feature,embedding_combine_formal,embedding_combine_latter,embedding_combine_random,embedding_combine_odd,embedding_combine_even, embedding_word, num_classes, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0001):
        self.embedding_size = embedding_size
        self.embedding = embedding
        self.embedding_word = embedding_word
        self.embedding_single_feature = embedding_single_feature
        self.embedding_combine_feature = embedding_combine_feature
        self.embedding_mark = embedding_mark
        self.embedding_combine_formal = embedding_combine_formal
        self.embedding_combine_latter = embedding_combine_latter
        self.embedding_combine_random = embedding_combine_random
        self.embedding_combine_odd = embedding_combine_odd
        self.embedding_combine_even = embedding_combine_even
        self.num_classes = num_classes
        self.dropout_keep_prob = 0.7
        initializer = tf.keras.initializers.glorot_normal
        l2_loss = tf.constant(0.0)

        # 创建邻接矩阵
        # node_num = 25
        # adj_feature = np.zeros([node_num, node_num], dtype=float)
        #
        # """
        # entity_Position   0
        # entity_1          1 2 3 4
        # entity_1_type     5
        # entity1_Subtype,  6
        # entity1_Head,     7 8 9 10
        # RightPos_Entity1, 11
        # LeftPos_Entity1,  12
        # entity_2,         13 14 15 16
        # entity_2_type,    17
        # entity2_Subtype,  18
        # entity2_Head,     19 20 21 22
        # RightPos_Entity2, 23
        # LeftPos_Entity2   24

        # Featute_1 = concate(RightPos_Entity1, Type_Entity1)
        # Featute_2 = concate(LeftPos_Entity1, Type_Entity1)
        # Featute_3 = concate(RightPos_Entity2, Type_Entity2)
        # Featute_4 = concate(LeftPos_Entity2, Type_Entity2)
        # Featute_5 = concate(Type_Entity1, Type_Entity2)
        # Featute_6 = concate(Subtype_Entity1, Subtype_Entity2)
        # Featute_7 = concate(Head_Entity1, Head_Entity2)

        # adj[0, 0], adj[1, 1], adj[2, 2], adj[3, 3], adj[4, 4], adj[5, 5], adj[6, 6], adj[7, 7], adj[8, 8], adj[9, 9], \
        # adj[10, 10], adj[11, 11], adj[12, 12], adj[13, 13], adj[14, 14], adj[15, 15], adj[16, 16], adj[17, 17], adj[
        #     18, 18], adj[19, 19], adj[20, 20], adj[21, 21], adj[22, 22], \
        # adj[23, 23], adj[24, 24] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        # ind_i = [[11],[12],[23],[24],[5],[6],[7,8,9,10]]
        # ind_j = [[5],[5],[17],[17],[17],[18],[19,20,21,22]]
        # for k,i_list in enumerate(ind_i):
        #     j_list = ind_j[k]
        #     for i in i_list:
        #         for j in j_list:
        #             adj_feature[i, j] = 1
        #             adj_feature[j, i] = 1
        #
        # adj = np.array(preprocess_adj(adj_feature))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj = tf.Variable(adj, dtype=tf.float32, name='adj', trainable=False)

        # 创建邻接矩阵
        # node_num = 100
        # adj_word = []
        # for i in range(node_num):
        #     x = np.random.normal(loc=1, scale=0.5, size=node_num)
        #     adj_word.append(np.array(x))
        # adj_word = np.array(adj_word)
        # # adj_word = np.random.rand(node_num, node_num)
        # # adj_word = np.random.normal(loc=0,scale=1e-2,size=shape)
        # adj_word = np.array(preprocess_adj(adj_word))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_word = tf.Variable(adj_word, dtype=tf.float32, name='adj_word', trainable=True)
        # #
        # node_num = 19
        # adj_feature = []
        # for i in range(node_num):
        #     x = np.random.normal(loc=1, scale=0.5, size=node_num)
        #     adj_feature.append(np.array(x))
        # adj_feature = np.array(adj_feature)
        # # adj_feature = np.random.rand(node_num, node_num)
        # adj_feature = np.array(preprocess_adj(adj_feature))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_feature = tf.Variable(adj_feature, dtype=tf.float32, name='adj_feature', trainable=True)
        #
        # node_num = 24
        # adj_com_feature = []
        # for i in range(node_num):
        #     x = np.random.normal(loc=1, scale=0.5, size=node_num)
        #     adj_com_feature.append(np.array(x))
        # adj_com_feature = np.array(adj_com_feature)
        # # adj_feature = np.random.rand(node_num, node_num)
        # adj_com_feature = np.array(preprocess_adj(adj_com_feature))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_com_feature = tf.Variable(adj_com_feature, dtype=tf.float32, name='adj_com_feature', trainable=True)
        #
        #
        # node_num = 4
        # adj_mark = []
        # for i in range(node_num):
        #     x = np.random.normal(loc=1, scale=0.5, size=node_num)
        #     adj_mark.append(np.array(x))
        # adj_mark = np.array(adj_mark)
        # # adj_mark = np.random.rand(node_num, node_num)
        # adj_mark = np.array(preprocess_adj(adj_mark))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_mark = tf.Variable(adj_mark, dtype=tf.float32, name='adj_mark', trainable=True)
        #
        # node_num = 19 + 24
        # adj_sing_com = []
        # for i in range(node_num):
        #     x = np.random.normal(loc=1, scale=0.5, size=node_num)
        #     adj_sing_com.append(np.array(x))
        # adj_sing_com = np.array(adj_sing_com)
        # # adj_mark = np.random.rand(node_num, node_num)
        # adj_sing_com = np.array(preprocess_adj(adj_sing_com))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_sing_com = tf.Variable(adj_sing_com, dtype=tf.float32, name='adj_sing_com', trainable=True)
        #
        # self.embedding_word = tf.Variable(embedding_word, dtype=tf.float32, name='embedding_word', trainable=True)
        # self.embedding_mark = tf.Variable(embedding_mark, dtype=tf.float32, name='embedding_mark', trainable=True)
        # self.embedding_single_feature = tf.Variable(embedding_single_feature, dtype=tf.float32,
        #                                             name='embedding_single_feature', trainable=True)
        # self.embedding_combine_feature = tf.Variable(embedding_combine_feature, dtype=tf.float32,
        #                                              name='embedding_combine_feature', trainable=True)

        # self.adj_com_feature = tf.Variable(tf.random.truncated_normal([24, 24], -1, 1),
        #                            name='adj_com_feature', dtype=tf.float32)

        # node_num = 12
        # adj_formal = np.random.rand(node_num, node_num)
        # adj_formal = np.array(preprocess_adj(adj_formal))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_formal = tf.Variable(adj_formal, dtype=tf.float32, name='adj_formal', trainable=True)
        #
        # node_num = 12
        # adj_latter = np.random.rand(node_num, node_num)
        # adj_latter = np.array(preprocess_adj(adj_latter))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_latter = tf.Variable(adj_latter, dtype=tf.float32, name='adj_latter', trainable=True)
        #
        # node_num = 12
        # adj_random = np.random.rand(node_num, node_num)
        # adj_random = np.array(preprocess_adj(adj_random))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_random = tf.Variable(adj_random, dtype=tf.float32, name='adj_random', trainable=True)

        # node_num = 131
        # adj_final = np.random.rand(node_num, node_num)
        # adj_final = np.array(preprocess_adj(adj_final))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
        # self.adj_final = tf.Variable(adj_final, dtype=tf.float32, name='adj_final', trainable=True)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-max-pooling-%s' % filter_size):
                conv_1 = tf.layers.conv1d(self.embedding_word, num_filters, filter_size, name='conv1d_1')
                # conv_1_Graph = GraphConvolution(self.embedding_word.shape[2],100,self.adj_word,self.embedding_word).out()
                conv_2 = tf.layers.conv1d(self.embedding_mark, num_filters, 2, name='conv1d_2')
                # conv_2_Graph = GraphConvolution(self.embedding_mark.shape[2],100,self.adj_mark,self.embedding_mark).out()
                # conv_3 = tf.layers.conv1d(self.embedding_single_feature, num_filters, 1, name='conv1d_3')
                # conv_3_Graph = GraphConvolution(self.embedding_single_feature.shape[2],100,self.adj_feature,self.embedding_single_feature).out()
                # conv_4 = tf.layers.conv1d(self.embedding_combine_feature, num_filters, 1, name='conv1d_4')
                # conv_4_Graph = GraphConvolution(self.embedding_combine_feature.shape[2], 100, self.adj_com_feature,
                #                                 self.embedding_combine_feature).out()
                # conv_4 = tf.layers.conv1d(self.embedding_feature, num_filters, filter_size + 1, name='conv1d_4')
                # conv_word = tf.concat([conv_1,conv_2,conv_4],1) # 64 198 100
                # conv_mark = tf.concat([conv_2,conv_2_Graph],1) # 64 8 100
                # conv_feature = tf.concat([conv_3,conv_3_Graph],1) # 64 50 100
                conv_combine_formal = tf.layers.conv1d(self.embedding_combine_formal, num_filters, filter_size,
                                                       name='embedding_combine_formal')
                # conv_combine_formal_Graph = GraphConvolution(self.embedding_combine_formal.shape[2],100,self.adj_formal,self.embedding_combine_formal).out()
                conv_combine_latter = tf.layers.conv1d(self.embedding_combine_latter, num_filters, filter_size, name='embedding_combine_latter')
                # conv_combine_latter_Graph = GraphConvolution(self.embedding_combine_latter.shape[2], 100,
                #                                              self.adj_latter, self.embedding_combine_latter).out()
                conv_combine_random = tf.layers.conv1d(self.embedding_combine_random, num_filters, filter_size, name='embedding_combine_random')
                # conv_combine_random_Graph = GraphConvolution(self.embedding_combine_random.shape[2], 100,
                #                                              self.adj_random, self.embedding_combine_random).out()

                # self.final_conv = tf.concat([conv_1, conv_2, conv_3, conv_combine_formal, conv_combine_latter, conv_combine_random,conv_3_Graph],1)
                # final_conv_Graph = GraphConvolution(self.final_conv.shape[2], 100, self.adj_final, self.final_conv).out()

                # conv_combine_formal = tf.concat([conv_combine_formal,conv_combine_formal_Graph],1)
                # conv_combine_latter = tf.concat([conv_combine_latter,conv_combine_latter_Graph],1)
                # conv_combine_random = tf.concat([conv_combine_random,conv_combine_random_Graph],1)
                # conv_combine_odd = tf.layers.conv1d(self.embedding_combine_odd, num_filters, 1,
                #                                        name='conv_combine_odd')
                # conv_combine_even = tf.layers.conv1d(self.embedding_combine_even, num_filters, 1,
                #                                        name='conv_combine_even')
                conv = tf.concat([conv_1, conv_2, conv_combine_formal,conv_combine_latter,conv_combine_random], 1) # 64 256 100
                # conv = tf.concat([final_conv_Graph],1)  # 64 256 100
                pooled = tf.reduce_max(conv, reduction_indices=[1], name='max-pooling')
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def pro(self):
        # return self.loss, self.losses, self.scores, self.probability
        return self.h_drop

# class CNN_model_conv():
#     def __init__(self, embedding, embedding_mark, embedding_feature, embedding_word, num_classes, embedding_size,
#                  filter_sizes, num_filters, l2_reg_lambda=0.0001):
#         self.embedding_size = embedding_size
#         self.embedding = embedding
#         self.embedding_mark = embedding_mark
#         self.embedding_feature = embedding_feature
#         self.embedding_word = embedding_word
#         self.num_classes = num_classes
#         self.dropout_keep_prob = 0.7
#         initializer = tf.keras.initializers.glorot_normal
#         l2_loss = tf.constant(0.0)
#
#         pooled_outputs = []
#         for i, filter_size in enumerate(filter_sizes):
#             with tf.variable_scope('conv-max-pooling-%s' % filter_size):
#                 conv_1 = tf.layers.conv1d(self.embedding_word, num_filters, 3, name='conv1d_1')
#                 conv_2 = tf.layers.conv1d(self.embedding_mark, num_filters, filter_size, name='conv1d_2')
#                 conv_3 = tf.layers.conv1d(self.embedding_feature, num_filters, 1, name='conv1d_3')
#                 # conv_4 = tf.layers.conv1d(self.embedding_feature, num_filters, filter_size + 1, name='conv1d_4')
#                 conv = tf.concat([conv_1, conv_2, conv_3], 1)
#                 pooled = tf.reduce_max(conv, reduction_indices=[1], name='max-pooling')
#                 pooled_outputs.append(pooled)
#
#         num_filters_total = num_filters * len(filter_sizes)
#         self.h_pool = tf.concat(pooled_outputs, 1)
#         self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
#
#         with tf.name_scope("dropout"):
#             self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
#
#     def pro(self):
#         # return self.loss, self.losses, self.scores, self.probability
#         return self.h_drop

# class GCN_model_():
#
#     def __init__(self, embedding, embedding_feature, num_classes, vocab_size,
#                  filter_sizes, num_filters,
#                  l2_reg_lambda=0.0001):
#         self.embedding = embedding
#         # self.embedding_multi = embedding_multi
#         # self.embedding_feature = embedding_feature
#         # self.embedding_mark = embedding_mark
#         self.num_classes = num_classes
#         self.dropout_keep_prob = 0.5
#         # with tf.device('/gpu:0'), tf.compat.v1.variable_scope('bert_lstm_out'):
#         #     lstm_layer = tf.keras.layers.LSTM(768, return_sequences=True, return_state=True)
#         #     self.embedding_sen, _, _ = lstm_layer(self.embedding_sen)
#
#         # with tf.compat.v1.variable_scope('_vars'):
#         #
#         #     self.weights = tf.Variable(tf.random.truncated_normal([self.embedding_sen.shape[2], 300], -0.35, 0.35),
#         #                                name='weights_se', dtype=tf.float32)
#         # self.embedding_sen = tf.matmul(self.embedding_sen, self.weights)
#         # with tf.device('/gpu:0'), tf.compat.v1.variable_scope("embedding_sen_dense"):
#         #     self.liner = tf.compat.v1.layers.dense(self.embedding_sen, 300, activation=tf.nn.relu)
#         # Embedding layer
#         # with tf.device('/gpu:0'), tf.compat.v1.variable_scope("atomic-embedding"):
#         #     self.W_text1 = tf.Variable(tf.compat.v1.random_uniform([vocab_size, atomic_embedding_size], -0.25, 0.25),
#         #                                name="W_text1_a")
#         #     self.text_embedded_atomic = tf.nn.embedding_lookup(self.W_text1, self.embedding_atomic, name='atomic')
#
#         # self.embedding = tf.concat([self.embedding_feature,self.embedding_word, self.embedding_multi], axis=1) # (embedding) 32 211 768
#         # self.embedding = self.embedding_feature
#         # with tf.device('/gpu:0'), tf.variable_scope('bert_lstm_out'):
#         #     lstm_layer = tf.keras.layers.LSTM(300, return_sequences=True, return_state=True)
#         #     self.h_pool, _, _ = lstm_layer(self.embedding)
#         # self.embedding = tf.concat([self.text_embedded_atomic, self.embedding_multi], axis=1)
#         # self.embedding = self.embedding_sen
#         # 创建邻接矩阵
#         node_num = 25
#         adj = np.zeros([node_num, node_num], dtype=float)
#         #
#         # """
#         # # entity_Position
#         # # entity_1
#         # # entity_2
#         # # RightPos_Entity1
#         # # LeftPos_Entity1
#         # # entity1_type
#         # # entity1_subtype
#         # # entity1_head
#         # # RightPos_Entity2
#         # # LeftPos_Entity2
#         # # entity2_type
#         # # entity2_subtype
#         # # entity2_head
#         # # mark1
#         # # mark2
#         # # mark3
#         # # mark4
#         # """
#         # adj[0, 0], adj[1, 1], adj[2, 2], adj[3, 3], adj[4, 4], adj[5, 5], adj[6, 6], adj[7, 7], adj[8, 8], adj[9, 9], \
#         # adj[10, 10], adj[11, 11], adj[12, 12], adj[13, 13], adj[14, 14], adj[15, 15], adj[16, 16], adj[17, 17], adj[
#         #     18, 18], adj[19, 19], adj[20, 20], adj[21, 21], adj[22, 22], \
#         # adj[23, 23], adj[24, 24] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
#         adj[1, 3], adj[3, 1] = 1, 1
#         adj[1, 4], adj[4, 1] = 1, 1
#         adj[1, 5], adj[5, 1] = 1, 1
#         adj[1, 6], adj[6, 1] = 1, 1
#         adj[1, 7], adj[7, 1] = 1, 1
#         adj[2, 8], adj[8, 2] = 1, 1
#         adj[2, 9], adj[9, 2] = 1, 1
#         adj[2, 10], adj[10, 2] = 1, 1
#         adj[2, 11], adj[11, 2] = 1, 1
#         adj[2, 12], adj[12, 2] = 1, 1
#         adj[3, 8], adj[8, 3] = 1, 1
#         adj[4, 9], adj[9, 4] = 1, 1
#         adj[5, 10], adj[10, 5] = 1, 1
#         adj[6, 11], adj[11, 6] = 1, 1
#         adj[7, 12], adj[12, 7] = 1, 1
#         # adj[13, 14], adj[14, 13] = 1, 1
#         # adj[15, 16], adj[16, 15] = 1, 1
#
#         adj = np.array(preprocess_adj(adj))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
#         self.adj = tf.Variable(adj, dtype=tf.float32, name='adj', trainable=True)
#         # “constant创建的是常数，tf. Variable创建的是变量。
#         # 变量属于可训练参数，在训练过程中其值会持续变化，也可以人工重新赋值，而常数的值自创建起就无法改变。”
#
#         # 随机初始化矩阵 参与训练 way 3
#         # adj = np.zeros([node_num, node_num], dtype=float)
#         # adj = np.random.rand(node_num, node_num)
#         # adj = np.array(preprocess_adj(adj))  # 计算D^(-1/2)*(A+I)*D^(-1/2)
#         # self.adj = tf.Variable(adj, dtype=tf.float32, name='adj', trainable=False)
#
#         with tf.device('/gpu:0'), tf.compat.v1.variable_scope("gcn-embedding"):
#             # 调用GCN (64, 13, 100)
#             # 一层 layer_out_1 (32 211 100) embedding (32 211 768)
#             # layer_out_1 = GraphConvolution(self.embedding.shape[2], 100, self.adj,
#             #                                self.embedding).out()
#
#             # 两层
#             layer_out_1 = GraphConvolution(self.embedding.shape[2], 100, self.adj,
#                                            self.embedding).out()
#             layer_out_2_in = tf.concat([layer_out_1, self.embedding], axis=-1)
#             # layer_out_2_in = tf.add(layer_out_1, self.embedding)
#             layer_out_2 = GraphConvolution(layer_out_2_in.shape[2], 100, self.adj,
#                                            layer_out_2_in).out()
#             layer_out_2 = tf.concat([layer_out_2,layer_out_2_in],axis=-1)
#
#         # (64,1300)
#         self.infoEmbbding = tf.reshape(layer_out_2, [-1, layer_out_2.shape[1] * layer_out_2.shape[2]])
#
#         self.h_pool_flat = self.infoEmbbding
#         with tf.compat.v1.name_scope("dropout"):
#             self.h_drop = tf.nn.dropout(self.h_pool_flat, 1 - (self.dropout_keep_prob))
#
#         # self.h_drop = tf.concat([self.infoEmbbding, self.h_drop], 1)
#
#     def pro(self):
#         # return self.loss, self.losses, self.scores, self.probability
#         return self.h_drop
#
# class CNN_model_():
#     def __init__(self,sentence,filter_sizes, num_filters, vocab_size, text_embedding_size,
#                  l2_reg_lambda=0.0001):
#         self.sentence = sentence
#         self.dropout_keep_prob = 0.7
#
#         # Embedding layer
#         with tf.device('/gpu:1'), tf.variable_scope('text-embedding'):
#             self.W_text = tf.Variable(tf.random_uniform([vocab_size,text_embedding_size],-0.25,0.25),name='W_text')
#             self.text_embedding = tf.nn.embedding_lookup(self.W_text,self.sentence,name = 'sentence')
#
#         # with tf.device('/gpu:1'), tf.variable_scope('lstm-pooling'):
#         #     rnn_layer = tf.keras.layers.LSTM(300,return_sequence=True, return_state=True)
#         #     self.h_pool,final_memory_state,final_carry_state = rnn_layer(self.text_embedding)
#
#         pooled_outputs = []
#         for i, filter_size in enumerate(filter_sizes):
#             with tf.variable_scope('conv-max-pooling-%s' % filter_size):
#                 # 2(64,93,60) 3(64,92,60) 4(64,91,60) 5(64,90,60)
#                 conv_1 = tf.layers.conv1d(self.text_embedding, num_filters, filter_size, name='conv1d_1',
#                                                     reuse=tf.AUTO_REUSE)
#                 # (64,50)
#                 # pooled = tf.reduce_max(input_tensor=conv_1, axis=[1], name='max-pooling')
#                 pooled_outputs.append(conv_1)
#
#         # num_filters_total = num_filters * len(filter_sizes)
#         self.h_pool = tf.concat(pooled_outputs, 1)
#         with tf.device('/gpu:0'), tf.compat.v1.variable_scope("embedding_sen_dense"):
#             self.h_pool = tf.layers.dense(self.h_pool, 300, activation=tf.nn.relu)
#         self.h_drop = self.h_pool
#         with tf.name_scope("dropout"):
#             self.h_drop = tf.nn.dropout(self.h_pool, 1 - (self.dropout_keep_prob))
#
#     def pro(self):
#         # return self.loss, self.losses, self.scores, self.probability
#         return self.h_drop


# 加载中文字向量wiki_100.txt
def load_word2vec(embedding_path, embedding_dim, vocab):

    # initial matrix with random uniform
    # initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    initW = np.random.randn(len(vocab), embedding_dim).astype(np.float32) / np.sqrt(len(vocab))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(embedding_path))
    with open(embedding_path, "r",encoding='utf-8' ) as f:
        lines = f.readlines()
        vocab_size = len(lines)
        for line in lines:
            word = line.split(' ')[0]
            embedding = ' '.join(line.split(' ')[1:])
            idx = vocab.get(word)
            if idx != 0:
                word_embedding = np.fromstring(embedding, dtype='float32',sep = ' ')
                initW[idx] = word_embedding
            else:
                continue
    return initW

# def load_word2vec(embedding_path, embedding_dim, vocab):
#     # initial matrix with random uniform
#     initW = np.random.randn(len(vocab), embedding_dim).astype(np.float32) / np.sqrt(len(vocab))
#     # load any vectors from the word2vec
#     print("Load word2vec file {0}".format(embedding_path))
#     with open(embedding_path, "rb" ) as f:
#         header = f.readline()
#         vocab_size, layer_size = map(int, header.split())
#         binary_len = np.dtype('float32').itemsize * layer_size
#         for line in range(vocab_size):
#             word = []
#             while True:
#                 ch = f.read(1).decode('latin-1')
#                 if ch == ' ':
#                     word = ''.join(word)
#                     break
#                 if ch != '\n':
#                     word.append(ch)
#             if word in vocab:
#                 idx = vocab[word]
#             else:
#                 idx = 0
#             if idx != 0:
#                 initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
#             else:
#                 f.read(binary_len)
#     return initW

def batch_gather(params, indexs, length):
    tensors = []
    for i in range(length):
        indices = indexs[:, i]
        range_ = tf.range(0, tf.shape(indices)[0])
        indices = tf.stack([range_, indices], 1)
        tensors.append(indices)
    indices = tf.stack(tensors, 1)
    return tf.gather_nd(params, indices)

# def create_model_gcn(bert_config, is_training, input_ids, input_mask, segment_ids,
#                  labels, num_labels, feature_index, # mark_index, word_index, left, entity1, mid, entity2, right, left_index, entity1_index, mid_index, entity2_index, right_index, all_features,
#                  use_one_hot_embeddings, tokenizer=None):
#     """Creates a classification model."""
#     model = modeling.BertModel(
#         config=bert_config,
#         is_training=is_training,
#         input_ids=input_ids,
#         input_mask=input_mask,
#         token_type_ids=segment_ids,
#         use_one_hot_embeddings=use_one_hot_embeddings)
#
#     # In the demo, we are doing a simple classification task on the entire
#     # segment.
#     #
#     # If you want to use the token-level output, use model.get_sequence_output()
#     # instead.
#
#     # output_layer1 = model.get_pooled_output()  # 主干模型获得的模型输出
#
#     # embedding = model.get_sequence_output()  # 原子特征 + 实体标记后的句子
#     cnn = CNN_model(sentence=input_ids, filter_sizes=[2,3,4,5], num_filters=60, vocab_size=30522, text_embedding_size=300)
#     pretrain_W = load_word2vec('./data/GoogleNews-vectors-negative300.bin', 300, vocab=tokenizer.vocab)
#     cnn.W_text.assign(pretrain_W)
#     embedding = cnn.text_embedding
#     print("111111111", embedding.shape)
#     # embedding_mark = batch_gather(embedding, mark_index, 4)
#     # print("111111111", embedding_mark.shape)
#     embedding_feature = batch_gather(embedding, feature_index, 13)
#     print("111111111", embedding_feature.shape)
#     # embedding_word = batch_gather(embedding, word_index, 100)
#     # print("111111111", embedding_word.shape)
#
#     # embedding_left = batch_gather(embedding, left_index, 30)
#     # print("111111111", embedding_left.shape)
#     # embedding_entity1 = batch_gather(embedding, entity1_index, 4)
#     # print("111111111", embedding_entity1.shape)
#     # embedding_mid = batch_gather(embedding, mid_index, 30)
#     # print("111111111", embedding_mid.shape)
#     # embedding_entity2 = batch_gather(embedding, entity2_index, 4)
#     # print("111111111", embedding_entity2.shape)
#     # embedding_right = batch_gather(embedding, right_index, 30)
#     # print("111111111", embedding_right.shape)
#
#     # embedding_multi = tf.concat([embedding_left, embedding_entity1,embedding_mid,embedding_entity2,embedding_right], axis=1)
#
#     # gcn = GCN_model(embedding_word, embedding, embedding_feature, embedding_mark, num_classes = num_labels, vocab_size = 30522,
#     #              l2_reg_lambda=0.0001)
#
#
#     # gcn = GCN_model(embedding_word=embedding_word, embedding_multi=embedding_multi, embedding_feature=embedding_feature,embedding_mark=embedding_mark,# embedding_atomic=all_atomic,
#     #                 num_classes=num_labels, vocab_size=30522, filter_sizes=[2, 3, 4, 5],
#     #                 num_filters=60)
#     gcn = GCN_model_(embedding=embedding,  embedding_feature=embedding_feature,  # embedding_atomic=all_atomic,
#                     num_classes=num_labels, vocab_size=30522, filter_sizes=[2, 3, 4, 5],
#                     num_filters=60)
#     # loss, per_example_loss, logits, probabilities = cnn.pro()
#     output_layer = gcn.pro()
#
#     hidden_size = output_layer.shape[-1].value  # bert的输出大小
#     # hidden_size1 = embedding.shape[-1].value
#
#     output_weights = tf.get_variable(
#         "output_weights", [num_labels, hidden_size],
#         initializer=tf.truncated_normal_initializer(stddev=0.02))
#
#     output_bias = tf.get_variable(
#         "output_bias", [num_labels], initializer=tf.zeros_initializer())
#
#     with tf.variable_scope("loss"):
#         if is_training:
#             # I.e., 0.1 dropout
#             output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
#
#         logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # 分类层
#         logits = tf.nn.bias_add(logits, output_bias)
#
#         """在此处改为多标签分类的sigmoid激活函数"""
#         # probabilities = tf.nn.sigmoid(logits)  # 该值是输出的预测结果
#         probabilities = tf.nn.softmax(logits)  # 该值是输出的预测结果
#         labels = tf.cast(labels, tf.float32)
#
#         per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
#         loss = tf.reduce_mean(per_example_loss)
#         return (loss, per_example_loss, logits, probabilities)

def create_model_cnn(bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, mark_index, single_feature_index,combine_feature_index,formal_12,latter_12,random_12,odd_number,even_number, word_index, num_labels, use_one_hot_embeddings,tokenizer=None):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    cnn = CNN_model(sentence=input_ids, filter_sizes=[1, 3, 5, 7, 9], num_filters=100, vocab_size=21128,
                    text_embedding_size=100)
    # pretrain_W = load_word2vec('./data/GoogleNews-vectors-negative300.bin', 300, vocab=tokenizer.vocab)
    pretrain_W = load_word2vec('./data/wiki_100.utf8', 100, vocab=tokenizer.vocab)
    # pretrain_W = load_word2vec('./data/wiki_100.utf8', 100, vocab=tokenizer.vocab)
    cnn.W_text.assign(pretrain_W)
    embedding = cnn.text_embedding

    # embedding = model.get_sequence_output()
    # embedding = model.get_pooled_output()
    print("111111111", embedding.shape)
    embedding_mark = batch_gather(embedding, mark_index, 4)
    # embedding_mark = tf.gather_nd(embedding, [[[0, 0], [0, 3], [0,5], [0,6]], [[1,2], [1,4], [1,2], [1,3]]])
    print("111111111", embedding_mark.shape)
    embedding_single_feature = batch_gather(embedding, single_feature_index, 21)
    print("111111111", embedding_single_feature.shape)
    embedding_combine_feature = batch_gather(embedding, combine_feature_index, 40)
    print("111111111", embedding_combine_feature.shape)
    embedding_word = batch_gather(embedding, word_index, 100)
    print("111111111", embedding_word.shape)
    embedding_combine_formal = batch_gather(embedding_combine_feature, formal_12, 20)
    print("111111111", embedding_combine_formal.shape)
    embedding_combine_latter = batch_gather(embedding_combine_feature, latter_12, 20)
    print("111111111", embedding_combine_latter.shape)
    embedding_combine_random = batch_gather(embedding_combine_feature, random_12, 20)
    print("111111111", embedding_combine_random.shape)
    embedding_combine_odd = batch_gather(embedding_combine_feature, odd_number, 20)
    print("111111111", embedding_combine_odd.shape)
    embedding_combine_even = batch_gather(embedding_combine_feature, even_number, 20)
    print("111111111", embedding_combine_even.shape)


    hidden_size = embedding.shape[-1].value

    cnn = GNN_model_conv(embedding=embedding, embedding_mark=embedding_mark, embedding_single_feature=embedding_single_feature,embedding_combine_feature=embedding_combine_feature,embedding_combine_formal=embedding_combine_formal,embedding_combine_latter=embedding_combine_latter,embedding_combine_random=embedding_combine_random,embedding_combine_odd=embedding_combine_odd,embedding_combine_even=embedding_combine_even,
                    embedding_word=embedding_word, num_classes=num_labels,
                    embedding_size=hidden_size, filter_sizes=[1, 3, 5, 7, 9], num_filters=100)

    # loss, per_example_loss, logits, probabilities = cnn.pro()
    output_layer = cnn.pro()
    hidden_size1 = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size1],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.7)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        labels = tf.to_float(labels)

        per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, tokenizer=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        mark_index = features["mark_index"]
        single_feature_index = features["single_feature_index"]
        combine_feature_index = features["combine_feature_index"]
        formal_12 = features["formal_12"]
        latter_12 = features["latter_12"]
        random_12 = features["random_12"]
        odd_number = features["odd_number"]
        even_number = features["even_number"]
        word_index = features["word_index"]
        #
        # left = features['left']
        # entity1 = features['entity1']
        # mid = features['mid']
        # entity2 = features['entity2']
        # right = features['right']
        #
        # left_index = features['left_index']
        # entity1_index = features['entity1_index']
        # mid_index = features['mid_index']
        # entity2_index = features['entity2_index']
        # right_index = features['right_index']
        #
        # all_atomic = features['all_atomic']
        # all_features = input_ids

        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        ####create_model是原始的bert加全连接，create_cnn_attention_model在bert后增加cnn和attention
        # (total_loss, per_example_loss, logits, probabilities) = create_model(
        #     bert_config, is_training, input_ids, input_mask, segment_ids,
        #     label_ids, num_labels, use_one_hot_embeddings)
        (total_loss, per_example_loss, logits, probabilities) = create_model_cnn(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, mark_index, single_feature_index,combine_feature_index,formal_12,latter_12,random_12,odd_number,even_number,
            word_index,num_labels, use_one_hot_embeddings,tokenizer=tokenizer)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.nn.sigmoid(logits)

                def multi_label_hot(predictions, threshold=0.5):
                    predictions = tf.cast(predictions, tf.float32)
                    threshold = float(threshold)
                    return tf.cast(tf.greater(predictions, threshold), tf.int64)

                one_hot_prediction = multi_label_hot(predictions)
                accuracy = tf.metrics.accuracy(tf.cast(one_hot_prediction, tf.int32), label_ids)
                # predictions = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(logits),0.5),tf.int32),tf.cast(label_ids,tf.int32))
                # accuracy = tf.reduce_mean(tf.cast(predictions,tf.float32))
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def result_labels(data_type):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "relation_extraction": Relation_Extraction,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    print(label_list)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        tokenizer=tokenizer)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file) # 把特征写进train_file,并在后边从文件中读取
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder( # 传入保存有特征的文件
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        # predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_examples = processor.get_test_examples(data_type)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        pred_labels = []
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            # output_file = './data/output.json'
            # output_file = open(output_file,'w')
            i = 0
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                # print(prediction)
                # print(type(prediction))
                prediction = prediction['probabilities']
                # print(prediction)
                # print(type(prediction))
                predicted_labels = []
                predicted_scores = []
                count = 0
                for i, score in enumerate(prediction):
                    if score >= 0.5:
                        predicted_scores.append(score)
                        predicted_labels.append(i)
                        count += 1
                pred_labels.append(predicted_labels)
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)
    return pred_labels

def generate_pred_file(tags_list,tags, inf_path, outf_path):
    # save_PRF = open('./output/save_PRF_' + str(PRF_name) + '.txt', 'a+', encoding='utf-8')
    with open(inf_path, 'r',encoding='utf-8') as inf,open(outf_path, 'w', encoding='utf-8') as outf:
        pred_labels = result_labels(inf_path)
        true_labels = []
        sum = 0
        count = 0
        # lines = inf.readlines()[0:1000]
        lines = inf.readlines()
        for line in lines:
            line = line.split(' ')
            sent = line[-2].strip()
            test_label = line[-1].strip()
            true_labels.append(tags[test_label])
            pred_label = pred_labels[sum]
            sum += 1
            label_names = []
            for label in pred_label:
                label_names.append(tags_list[label])
            relation = label_names
            outf.write(sent + ' ' + str(relation) + '\n')

        preds = []
        for labels in pred_labels:
            if len(labels) != 0:
                preds.append(labels[0])
            else:
                preds.append(0)
        # preds = pred_labels
        truths = true_labels
        target_names = []
        labels_num = []
        for label in tags:
            if label == 'Negative':
                continue
            target_names.append(label)
        for i in range(1, len(tags)):
            labels_num.append(i)
        cls_results = metrics.classification_report(truths, preds, labels=labels_num,
                                                    digits=4,
                                                    target_names=target_names)
        # print(cls_results)
        wr_line = open('./output/PRF.txt', 'a+', encoding='utf-8')
        check_p_name = open('./output/models/checkpoint', 'r', encoding='utf-8').readline()
        wr_line.write(check_p_name + '\n')
        wr_line.write(cls_results)
        wr_line.close()
        wr_line = open('./output/this_PRF.txt', 'w', encoding='utf-8')
        wr_line.write(cls_results)
        wr_line.close()
        lines = open('./output/this_PRF.txt', 'r', encoding='utf-8').readlines()
        f1 = 0
        for line in lines:
            print(line)
            if 'macro avg' in line:
                element_list = line.split('    ')
                p = float(element_list[1])
                r = float(element_list[2])
                f1 = (2 * p * r) / (p + r)
                # element_list = line.split('\t')[0].split('    ')[1].strip()
                # f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(
                #     line.split('\t')[0].split('    ')[2].strip()) / (
                #              float(line.split('\t')[0].split('    ')[1].strip()) + float(
                #          line.split('\t')[0].split('    ')[2].strip()))
        print('The True F1 of ' + check_p_name + ' is ', f1)


def count_max_f(file):
    lines = open(file,'r',encoding='utf-8').readlines()
    max_f = 0
    line_count = 0
    max_line_count = 0
    k = -14

    for line in lines:
        line_count += 1

        if 'macro avg' in line:
            element_list = line.split('    ')
            p = float(element_list[1])
            r = float(element_list[2])
            f1 = (2 * p * r) / (p + r)
            # f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(line.split('\t')[0].split('    ')[2].strip()) / (float(line.split('\t')[0].split('    ')[1].strip()) + float(line.split('\t')[0].split('    ')[2].strip()))
            if(f1 > max_f):
                max_f = f1
                max_line_count = line_count

    for i in range(15):
        print(lines[max_line_count + k])
        k += 1

    print('The true max f1:',max_f)
    print('最大值所在行占总行数的：', max_line_count / len(lines))



if __name__ == "__main__":
    wr_line = open('./output/PRF.txt', 'w', encoding='utf-8')
    wr_line.close()
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")

    start_time = datetime.datetime.now()
    tags_list = []
    tags = dict()
    i = 0
    flag = 0
    inf_path = os.path.join(FLAGS.data_dir, 'test.txt')
    outf_path = './output/__Predict_test.txt'
    with open(os.path.join(FLAGS.data_dir, 'tags.txt'), 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
            tags[line.strip()] = i
            i += 1
    generate_pred_file(tags_list,tags,inf_path, outf_path)
    count_max_f('./output/PRF.txt')

