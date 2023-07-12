"""
Using LDA to label the topic. There are 3 steps:
1. Use LDA to do topic modelling by setting the number of topics to 24, which the number of pre-defined topics as given in the test.
2. Extract keywords for each of the 24 topics obtained by LDA, and find the most similar pre-defined topic.
3. Mapping each topic from LDA to the most similar pre-defined topic.
"""

import pandas as pd
from config import *
import jieba
from gensim import corpora
from gensim.models.ldamodel import LdaModel


# Text preprocessing
def clean(doc, stopwords):
    words = [word for word in jieba.cut(doc) if word not in stopwords]
    return " ".join(words)


df = pd.read_csv(data_path)
docs = df['RAW_CONTENT'].tolist()

# Read Chinese stopwords
fp = open(stopwords_cn, "r", encoding="utf-8")
stp_cn = fp.readlines()
fp.close()
for i in range(len(stp_cn)):
    stp_cn[i] = stp_cn[i].rstrip("\n")

# Tokenize the context and remove stopwords
doc_clean = [clean(doc, stp_cn).split() for doc in docs]

# Preparing Document-Term Matrix
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the LDA model
ldamodel = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)

# Print the topics
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)

# Assign the documents to topics
doc_topics = [max(ldamodel.get_document_topics(doc), key=lambda x: x[1])[0] for doc in doc_term_matrix]
