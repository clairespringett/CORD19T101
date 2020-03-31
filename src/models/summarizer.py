import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import tokenization
from bert import run_classifier
from bert import optimization
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
from utils.stats_utils import *
from utils.nlp_utils import *

class summarizer(object):
    def __init__(self):
        
        self.ABBREVIATION = ["Sr.", "i.e."]
        self.SKIP = []
        self.BERT_URL = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
        self.bert_module = hub.Module(self.BERT_URL)   
    
    # return a tuple (text, indicator, gap_score). 
    # If indicator is false, no best k was found. 
    # If gap_score was not positive, summarization is not valid. 
    
    def smr_base(self, paragraph):
        if not paragraph or type(paragraph) is not str: 
            return paragraph, False, None
        paragraph = self.skip_vocab(paragraph)
        paragraph = self.replace_vocab(paragraph)
        sentenceList = paragraph_to_sentenceList(paragraph)
        embeddings = self.bert_embeddings(sentenceList)
        summary_index, success, gap_score = self.KMean_optimizer(embeddings)
        if success:
            return ".".join([sentenceList[i] for i in summary_index]), success, gap_score
        return paragraph, False, None             
    
    def KMean_optimizer(self, embeddings):    
        X_df = pd.DataFrame(embeddings)
        try:
            gap = gap_statistic(X_df, max_k = X_df.shape[0])
            kneedle3 = KneeLocator(list(gap.index), gap , S=1.0, curve='concave', direction='increasing')
            nk = kneedle3.knee
            km = KMeans(n_clusters=nk)
            km = km.fit(embeddings)    
            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
            gap_score = round(gap.loc[nk,],3)  

            if gap_score > 0:

                return closest, True, gap_score
            else:
                return -1, False, None                
        except:
            return -1, False, None  
         
    def bert_embeddings(self, sentences):
        # input: a list of string
        #module = self.bert_module
        tokenizer = self.create_tokenizer_from_hub_module()

        input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences = sentences, \
                                                            max_seq_len = 128, \
                                                            tokenizer = tokenizer)
        input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
        segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)

        bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        out = sess.run(bert_outputs, feed_dict={input_ids: input_ids_vals, \
                                                input_mask: input_mask_vals, segment_ids: segment_ids_vals})

        return out['pooled_output']


    def create_tokenizer_from_hub_module(self):
        tokenization_info = self.bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def skip_vocab(self, paragraph):
        for word in self.SKIP:
            paragraph = [s.replace(word, "") for s in paragraph]
        return paragraph

    def replace_vocab(self, paragraph):
        for word in self.ABBREVIATION:
            cleaned = "".join(word.split("."))
            paragraph = paragraph.replace(word, cleaned)
        return paragraph 

# wrapper functions
def smr_str(paragraph):
    if type(paragraph) is not str:
        raise TypeError("Input has to be a string.")
    smr = summarizer()
    return smr.smr_base(paragraph)

def smr_pdSeries(pdSeries):
    if type(pdSeries) is not pd.core.series.Series:
        raise TypeError("Input has to be a pandas series.")
    smr = summarizer()
    pooled = pdSeries.apply(smr.smr_base).apply(pd.Series)
    pooled.columns=['Summary','Success', "Gap_score"]
    return pooled
    