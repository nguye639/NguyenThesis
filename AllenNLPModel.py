import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from textblob import TextBlob
import spacy

import allennlp_models
from allennlp_models.classification import biattentive_classification_network
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.text_classifier import TextClassifierPredictor
import allennlp_models.tagging
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/stanford-sentiment-treebank-roberta.2021-03-11.tar.gz")

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

nlp = spacy.load("en_core_web_sm")
def cleanData(messages):
    for i in range(0,len(messages)):
        try:
            doc = nlp(messages.iloc[i])
            token_list = [token for token in doc]
            filtered_tokens = [token for token in doc if not token.is_stop]
            lemmas = [token.lemma_ for token in filtered_tokens]
            s = ' '.join(lemmas)
            
            messages.iloc[i] = re.sub(r'\W+', ' ', s)
        except:
            messages.iloc[i] = ""
        
        #if is_ascii(messages.iloc[i]) == False:
            # messages.iloc[i] = ""

    return messages

def useAllen(comment):
    try:
        messages = comment.split(".")
        messages = [x for x in messages if x]
    
        scores = []

        for message in messages:
            prediction = predictor.predict(message)
            score = (int(prediction["label"]) * 2 - 1) * max(prediction["probs"])
            scores.append(score)
    except:
        scores = 0
        
    return np.mean(scores)
