#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-07 11:58:24
# @Author  : Ziyi Zhao (zzhao37@syr.edu)
# @Version : 1.8
# @To Do List : find suitable default topic number and minimum topic probability
#               random seed impact the topic distribution and the recommend result
#               multiple models to generate average recommendation result

import os
import nose
import numpy
import scipy
import gensim
import logging
import math
import sys 
import string
from nltk.corpus import stopwords
from scipy.spatial import distance
from collections import OrderedDict
from operator import itemgetter
# for word tense
from nltk.stem import PorterStemmer, WordNetLemmatizer
# for TF-IDF
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

documents=["Apple is releasing a new product",
	        "Amazon sells many things",
	        "Microsoft announces Nokia acquisition",
	        "Congratulations Your line is upgrade eligible",
	        "Please proceed to see the upgrade , options for this line",
	        "I am trying to understand how gensim package in Python implements Latent Dirichlet Allocation",
	        "Is 1000 it providing with a probability of the occurrence of each word",
	        "there are now some very nice visualization tools for gaining an intuition of LDA using gensim",
	        "To prepare for similarity queries , we need to enter all documents which we want to compare against subsequent queries",
	        "Its mission is to help NLP practicioners try out popular topic modelling algorithms on large datasets easily , and to facilitate prototyping of new algorithms for researchers",
	        "Human machine interface for lab abc computer applications",
	        "A survey of user opinion of computer system response time",
	        "Relation of user perceived response time to error measurement",
	        "The generation of random binary unordered trees",
	        "Graph minors IV Widths of trees and well quasi ordering",
	        "Palgrave Macmillan and the journals editorial teams have selected papers from the archives of the journals to give a representative sample of the best of our content",
	        "As you can see , this compacts the whole thing into a piece of code managed entirely by the while loop",
	        "As a programmer , it is up to you which style to use but always remember that readability is important , and that speed is also importan"]

test_document=["Parthian king Gondophares ref Journal of the Numismatic Society of India 1968 vol 30 p 188 190 A K Narain ref This is however unlikely as this king was probably much earlier but he could have been one of the later Indo Parthian kings who were also named Gondophares Notes references Category Indo Parthian kingdom Category 1st century Iranian people Category Zoroastrian dynasties and rulers"]

DEFAULT_TOPIC_NUMBER=200

SOME_FIXED_SEED=40

DEFAULT_TOP_RANK=1

DEFAULT_TFIDF_SCORE=0.01

DEFAULT_TRAIN_PASSES=20

DEFAULT_DATASET='wiki'

###############################################################################################################################################
# process train dataset
def loadtraindataset(traindataset,argv):
# initialize logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	with open(argv[1]) as infile:
		for line in infile:
			traindataset.append(line)

	count=1
	#for doc in traindataset:
		#print 'Document',count,":"
		#print(doc)
		#print('\n')
		#count+=1

# remove the stop words
	print('Process stop words')
# restore into a set to improve the speed
	stop_word_set=set(stopwords.words('english'))
	texts_without_stop = [[word_with_stop for word_with_stop in document.lower().split() if word_with_stop not in stop_word_set] for document in traindataset]
	#print(texts_without_stop)

# remove the punctuation
	exclude=set(string.punctuation)
	texts_without_punctuation = [[word_with_punctuation for word_with_punctuation in text_without_stop if word_with_punctuation not in (exclude)] for text_without_stop in texts_without_stop]
	print('Process punctuation')
	#print(texts_without_punctuation)	

# remove the number
	print('Process digit')
	texts_without_digit= [[word_with_digit for word_with_digit in text_without_punctuation if not word_with_digit.isdigit()] for text_without_punctuation in texts_without_punctuation]
    #print(texts_without_digit)

# remove word tense
	print('Process tense and plural')
	texts_without_tense=[]
	for text_without_digit in texts_without_digit:
		temp=[]
		for word in text_without_digit:
			temp.append(PorterStemmer().stem(word.decode('utf-8')))
		texts_without_tense.append(temp)
	#print(texts_without_tense)

# prepare dataset for TF-IDF processing
	print('TF-IDF process')
	temp_dataset_for_tfidf=[]
	#temp_count=0
	for text_without_tense in texts_without_tense:
		temp_str=''
		for i in xrange(0,len(text_without_tense)-2):
			temp_str+=text_without_tense[i]
			temp_str+=' '
		temp_str+=text_without_tense[len(text_without_tense)-1]
		#print(temp_count)
		temp_dataset_for_tfidf.append(temp_str)
		#temp_count+=1
	#print temp_dataset_for_tfidf

# TF-IDF process
	vectorizer=CountVectorizer()
	transformer=TfidfTransformer()
	tfidf=transformer.fit_transform(vectorizer.fit_transform(temp_dataset_for_tfidf))
	word=vectorizer.get_feature_names()

	word_dict=dict()
	word_count=0
	for element in word:
		#print element,":",word_count
		#print('\n')
		word_dict.update({element:word_count})
		word_count+=1

	weight=tfidf.toarray()

	tfidf_result='./dataset_mix_match/tf_idf_each_doc_'
	tfidf_result+=str(DEFAULT_DATASET)
	tfidf_result+='_'
	tfidf_result+=str(DEFAULT_TOPIC_NUMBER)
	tfidf_result+='_'
	tfidf_result+=str(DEFAULT_TRAIN_PASSES)
	tfidf_result+='.txt'

	tfidffile=open(tfidf_result,'w')
	print(len(word))
	for i in range(len(weight)):
		#print "Doc:",i
		tfidffile.write("Doc:%s\n" % i)
		for j in range(len(word)):
			if weight[i][j]>0.01:
				#print word[j],weight[i][j]
				tfidffile.write(str(word[j]))
				tfidffile.write(' ')
				tfidffile.write(str(weight[i][j]))
				tfidffile.write('\n')

# remove the word with low TF-IDF score
	texts_after_tfidf=[]
	print('Remove word based on TF-IDF')
	doc_count=0
	for text_without_tense in texts_without_tense:
		print "Process DOC:",doc_count
		temp_in_tfidf=[]
		for item in text_without_tense:
			if check_tfidf(doc_count,word_dict,weight,item,DEFAULT_TFIDF_SCORE):
				temp_in_tfidf.append(item)
		texts_after_tfidf.append(temp_in_tfidf)
		doc_count+=1

# generate final dataset
	texts=texts_after_tfidf

	proceed_dataset_result='./dataset_mix_match/proceed_dataset_'
	proceed_dataset_result+=str(DEFAULT_DATASET)
	proceed_dataset_result+='_'
	proceed_dataset_result+=str(DEFAULT_TOPIC_NUMBER)
	proceed_dataset_result+='_'
	proceed_dataset_result+=str(DEFAULT_TRAIN_PASSES)
	proceed_dataset_result+='.txt'

	thefile=open(proceed_dataset_result,'w')
	for item in texts:
		thefile.write("%s\n" % item)
	#print(texts)

# save into dictionary
	dictionary=gensim.corpora.Dictionary(texts)
	#print(dictionary)

	dictionary_result='./dictionary&corpus_mix_match/processed_dictionary_'
	dictionary_result+=str(DEFAULT_DATASET)
	dictionary_result+='_'
	dictionary_result+=str(DEFAULT_TOPIC_NUMBER)
	dictionary_result+='_'
	dictionary_result+=str(DEFAULT_TRAIN_PASSES)
	dictionary_result+='.dict'

	dictionary.save(dictionary_result)

# convert document into bag-of-word format
	corpus=[dictionary.doc2bow(text) for text in texts]
	#print(corpus)
# stores index as well, allowing random access to individual documents
	corpus_result='./dictionary&corpus_mix_match/processed_corpus_'
	corpus_result+=str(DEFAULT_DATASET)
	corpus_result+='_'
	corpus_result+=str(DEFAULT_TOPIC_NUMBER)
	corpus_result+='_'
	corpus_result+=str(DEFAULT_TRAIN_PASSES)
	corpus_result+='.mm'
	gensim.corpora.MmCorpus.serialize(corpus_result,corpus)
    
# return corpus and dictionary
	return texts, corpus, dictionary

###############################################################################################################################################
# check the TF-IDF score
def check_tfidf(currentdoc,worddict,tfidfdict,currentword,score):
	try:
		#index=worddict.index(currentword)
		index=worddict.get(currentword,None)
		if index!=None:
			if tfidfdict[currentdoc][index]>score:
				return True
			return False
		#if tfidfdict[currentdoc][index]>score:
			#return True
	except ValueError:
		return False
	#for index,word in worddict.items():
		#if 	word==currentword:
			#if 	tfidfdict[currentdoc][index]>score:
				#return True	
			#return False
	return False

###############################################################################################################################################
# train lda model	
def trainlda(processed_corpus,processed_dictionary):
# initialize logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train lda model   # update with new documents# lda.update(other_corpus)
	lda=gensim.models.ldamodel.LdaModel(corpus=processed_corpus,id2word=processed_dictionary,num_topics=DEFAULT_TOPIC_NUMBER,update_every=1,chunksize=10000,passes=DEFAULT_TRAIN_PASSES,minimum_probability=0)

# save lda model
	model_name='./model_mix_match/ldamodel_'
	model_name+=str(DEFAULT_DATASET)
	model_name+='_'
	model_name+=str(DEFAULT_TOPIC_NUMBER)
	model_name+='_'
	model_name+=str(DEFAULT_TRAIN_PASSES)
	lda.save(model_name)
# return lda model for other functions to use it
	return lda

###############################################################################################################################################
# train word2vec model: in order to convert the word from text to vector
def trainword2vec(texts):
# initialize logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train word2vec model use the pre-processed text dataset (proceed by loadtraindataset())	
	word2vec_model = gensim.models.Word2Vec(texts, min_count=1) # default value is 5

# save trained word2vec model
	word2vec_model.save('word2vec_model')

# return word2vec model
	return word2vec_model

###############################################################################################################################################
# load dataset and train lda model
def loadnewdocument(new_document,dictionary):
# remove the stop words
	stop_word_set=set(stopwords.words('english'))
	new_texts_without_stop = [[new_word_with_stop for new_word_with_stop in new_document.lower().split() if new_word_with_stop not in stop_word_set] for new_document in new_document]
	#print(new_texts_without_stop)

# remove the punctuation
	exclude=set(string.punctuation)
	new_texts_without_punctuation = [[new_word_with_punctuation for new_word_with_punctuation in new_text_without_stop if new_word_with_punctuation not in (exclude)] for new_text_without_stop in new_texts_without_stop]
	#print(new_texts_without_punctuation)	

# remove the number
	new_texts_without_digit= [[new_word_with_digit for new_word_with_digit in new_text_without_punctuation if not new_word_with_digit.isdigit()] for new_text_without_punctuation in new_texts_without_punctuation]
    #print(new_texts_without_digit)

# generate final dataset
	new_texts_without_tense=[]
	for new_text_without_digit in new_texts_without_digit:
		new_temp=[]
		for word in new_text_without_digit:
			new_temp.append(PorterStemmer().stem(word.decode('utf-8')))
		new_texts_without_tense.append(new_temp)

# generate final dataset
	new_texts=new_texts_without_tense
	#print(new_texts)

# save into dictionary
	#new_dictionary=gensim.corpora.Dictionary(new_texts)
	#print(new_dictionary)

# convert document into bag-of-word format
	new_corpus=[dictionary.doc2bow(new_text) for new_text in new_texts]

	return new_corpus

###############################################################################################################################################
# load trained word2vec model
def loadword2vec(word2vec_model):
# load word2vec model
	load_word2vecmodel=gensim.models.Word2Vec.load('word2vec_model')
	#print(load_word2vecmodel.similarity('package', 'upgrade'))

###############################################################################################################################################
# infer the topic distribution for new document
def infertopics(ldamodel,new_document,dictionary):
# use new document and old dictionary to get new corpus
	new_corpus=loadnewdocument(new_document,dictionary)
	#print(new_corpus)

# load trained lda model
	load_ldamodel=gensim.models.ldamodel.LdaModel.load('ldamodel')

# get topic distributions
	topic_distribution=load_ldamodel.get_document_topics(new_corpus,0) # get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

# print topic distributions on new documents
	#for vector in topic_distribution:
		#print(vector)
	return topic_distribution

###############################################################################################################################################
# calculate the simliarity between new document and all old documents
def calculatesimlarity(new_doc_topic_distribution,new_document,old_documents,ldamodel): #dist = sqrt((xa-xb)^2 + (ya-yb)^2 + (za-zb)^2)
# print current new aeticle
	#print(new_document)

# initialize list to store probability
	old_document_topic_distribution = []
	new_document_topic_probability = []
	rank_list = dict()

# print and save new document topic distributions probability 
	for new_vector in new_doc_topic_distribution:
		for new_item in new_vector:
			#print("Topic:")
			#print(new_item[0])
			#print("Probability:")
			#print(new_item[1])
			new_document_topic_probability.append(new_item[1])
	#print(new_document_topic_probability)

# load trained lda model
	load_ldamodel=gensim.models.ldamodel.LdaModel.load('ldamodel')

# save old document topic distributions probability
	count=0
	for old_doc_corpus in old_documents:
		#print(old_doc_corpus)
		current_distribution=load_ldamodel.get_document_topics(old_doc_corpus,0) # get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
		#print "Document",count,":"
		#print(current_distribution)
		#print('\n')
		count +=1
		tempvec=[]
		for current_distribution_vec in current_distribution:
			#print(current_distribution_vec[1])
			tempvec.append(current_distribution_vec[1])
		old_document_topic_distribution.append(tempvec)		
	#print(old_document_topic_distribution)

# calculate the root mean square among each documents 
	doc_count=0
	for each_doc_topic_distribution in old_document_topic_distribution:
		#print(each_doc_topic_distribution)
		dst=distance.euclidean(new_document_topic_probability,each_doc_topic_distribution)
		#dst=calculate_euclidean_distance(default_topic_number,new_document_topic_probability,each_doc_topic_distribution)
		#print(dst)
		# save doc index and RMSE
		rank_list.update({doc_count:dst})
		doc_count+=1
		rank_list=OrderedDict(sorted(rank_list.items(),key=lambda t:t[1]))
	#print(rank_list)
	return rank_list

###############################################################################################################################################
# calculate the euclidean distance (dist = sqrt((xa-xb)^2 + (ya-yb)^2 + (za-zb)^2)
def calculate_euclidean_distance(default_topic_number,new_doc_dis,old_doc_dis):
	result=0
	count=0
	while count<DEFAULT_TOPIC_NUMBER:
		temp=0
		temp=math.pow((new_doc_dis[count]-old_doc_dis[count]),2)
		#print(temp)
		result+=temp
		#print(result)
		count+=1
	result=math.sqrt(result)
	#print(result)
	return result

###############################################################################################################################################
# recommend according to the rank list
def recommendation(rank_list,old_documents):
# print the whole rank list
	#for each_doc in rank_list.items():
		#print(each_doc)
	count=1
	temp_list=[]
	temp_list=list(rank_list)[:DEFAULT_TOP_RANK]

# print the default top rank of document index
	#print(temp_list)

# print the recommendation result
	print("\n")
	for each_item in temp_list:
		print "Top",count,"recommendation: doc",each_item+1
		#print(traindataset[each_item])
		print("\n")
		count+=1

###############################################################################################################################################
# main function
if __name__ == '__main__' :

	traindataset=[]
	rank_list=[]
	numpy.random.seed(SOME_FIXED_SEED)

# get 3 type of corpus
	texts, corpus, dictionary=loadtraindataset(traindataset,sys.argv)
# get word2vec model
	word2vec_model=trainword2vec(texts)
# get lda model
	ldamodel=trainlda(corpus,dictionary)
# infer topics for new document
	new_doc_topic_distribution=infertopics(ldamodel,test_document,dictionary)
# load word2vec model
	loadword2vec(word2vec_model)
# calculate the similarity
	rank_list=calculatesimlarity(new_doc_topic_distribution,test_document,corpus,ldamodel)
# recommendation by rank list	
	recommendation(rank_list,traindataset)
