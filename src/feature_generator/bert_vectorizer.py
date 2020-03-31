from dotenv import load_dotenv
load_dotenv()

import os
system_path=os.getenv("sys_path")

import sys 
sys.path.append(system_path) #this will allow us to define where to find modules 


from conf.local import config
import warnings
import pandas as pd

import tensorflow_hub as hub
import tensorflow as tf
import bert
from bert import tokenization
from bert import run_classifier


class BertFeatureGenerator:
    # class InputTypeEnum(Enum) :
    #     has_label = 1
    #     no_label = 2

    def __init__(self,max_seq_length):
        self.bert_url = config.bert_url
        self.bert_module = hub.Module (self.bert_url)        
        self.max_seq_length = max_seq_length
            
    def __create_tokenizer_from_hub_module(self):       
        tokenization_info = self.bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], \
                tokenization_info["do_lower_case"]])
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def generate_feature_from_batch_text_example(self,data,text_column_name, label_column_name):
        #text = data[text_column_name]
        label = data[label_column_name]
        input_examples = data.apply(lambda x: run_classifier.InputExample(guid=None, text_a=x[text_column_name], \
            text_b=None, label=x[label_column_name]), axis=1)
        
        tokenizer = self.__create_tokenizer_from_hub_module()
        label_list = pd.Categorical(label)
        label_list = label_list.codes 
        features = bert.run_classifier.convert_examples_to_features(input_examples, label_list, self.max_seq_length,\
            tokenizer)
        return (features)

