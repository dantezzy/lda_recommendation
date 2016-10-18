#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-19 13:54:24
# @Author  : Ziyi Zhao (zzhao37@syr.edu)
# @Version : 1.0

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

sys.path.insert(0, '/home/ziyizhao/lda_recommendation')
from lda_recommendation import loadtraindataset
from lda_recommendation import trainlda
from lda_recommendation import infertopics
from lda_recommendation import calculatesimlarity
from lda_recommendation import loadnewdocument

DEFAULT_TRAIN_DATASET='/home/ziyizhao/lda_recommendation/mix_match_experiment/dataset_mix_match/mix_match_train_dataset.txt'

DEFAULT_FIRST_TEST_DATASET='/home/ziyizhao/lda_recommendation/mix_match_experiment/dataset_mix_match/mix_match_test_dataset_first.txt'

DEFAULT_FIRST_TEST_DATASET_50='/home/ziyizhao/lda_recommendation/mix_match_experiment/dataset_mix_match/mix_match_test_dataset_first_50.txt'

DEFAULT_SECOND_TEST_DATASET='/home/ziyizhao/lda_recommendation/mix_match_experiment/dataset_mix_match/mix_match_test_dataset_second.txt'

DEFAULT_FLICKER_TRAIN='/home/ziyizhao/lda_recommendation/small_sentence_DataSet/flicker/flicker_train_dataset.txt'

DEFAULT_FLICKER_FIRST='/home/ziyizhao/lda_recommendation/small_sentence_DataSet/flicker/flicker_test_first.txt'

DEFAULT_FLICKER_SECOND='/home/ziyizhao/lda_recommendation/small_sentence_DataSet/flicker/flicker_test_second.txt'

DEFAULT_DATASET='wiki'

DEFAULT_TOPIC_NUMBER=200

SOME_FIXED_SEED=40

DEFAULT_TOP_RANK=1

DEFAULT_TRAIN_PASSES=20

###############################################################################################################################################
# run mix and match 
def run_mix_match():

	traindataset_test=[]
	traindataset_2=[]
	numpy.random.seed(SOME_FIXED_SEED)
	inputfile=[]
	inputfile.append('NULL')
	inputfile.append(DEFAULT_TRAIN_DATASET)
	#inputfile.append(DEFAULT_FLICKER_TRAIN)

	secondpart=[]
	secondpart.append('NULL')
	secondpart.append(DEFAULT_SECOND_TEST_DATASET)
	#secondpart.append(DEFAULT_FLICKER_SECOND)

# initialize list to store probability
	old_document_topic_distribution = []
	current_doc_index=0

# accuracy of the correct doc
	check_count=0
	check_count_10=0
	check_count_20=0

# accuracy of the correct tag
	tag_count_1=0
	tag_count_2=0
	tag_count_3=0

# count of each length doc
	doc_len_le_200=0
	doc_len_200_to_500=0
	doc_len_ge_500=0

# count accuracy of each length doc for top1
	correct_doc_le_200_top1=0
	correct_doc_200_to_500_top1=0
	correct_doc_ge_500_top1=0

# count accuracy of each length doc for top2
	correct_doc_le_200_top2=0
	correct_doc_200_to_500_top2=0
	correct_doc_ge_500_top2=0

# count accuracy of each length doc for top3
	correct_doc_le_200_top3=0
	correct_doc_200_to_500_top3=0
	correct_doc_ge_500_top3=0

# get 3 type of corpus for training dataset
	#print('train dicitonary')
	#texts, corpus, dictionary=loadtraindataset(traindataset_test,inputfile)

# load pre-processed dictionary and corpora
	print('Load proceed dicitonary')
	dictionary_result='./dictionary&corpus_mix_match/processed_dictionary_'
	dictionary_result+=str(DEFAULT_DATASET)
	dictionary_result+='_'
	dictionary_result+=str(DEFAULT_TOPIC_NUMBER)
	dictionary_result+='_'
	dictionary_result+=str(DEFAULT_TRAIN_PASSES)
	dictionary_result+='.dict'	
	dictionary=gensim.corpora.dictionary.Dictionary.load(dictionary_result)

	print('Load proceed corpus')
	corpus_result='./dictionary&corpus_mix_match/processed_corpus_'
	corpus_result+=str(DEFAULT_DATASET)
	corpus_result+='_'
	corpus_result+=str(DEFAULT_TOPIC_NUMBER)
	corpus_result+='_'
	corpus_result+=str(DEFAULT_TRAIN_PASSES)
	corpus_result+='.mm'
	corpus=gensim.corpora.MmCorpus(corpus_result)

# train lda model
	#print('train dicitonary')
	#ldamodel=trainlda(corpus,dictionary)

# load pre-trained lda model
	print('Load trained model')
	model_name='./model_mix_match/ldamodel_'
	model_name+=str(DEFAULT_DATASET)
	model_name+='_'
	model_name+=str(DEFAULT_TOPIC_NUMBER)
	model_name+='_'
	model_name+=str(DEFAULT_TRAIN_PASSES)
	ldamodel=gensim.models.ldamodel.LdaModel.load(model_name)

	topic_result='./result_mix_match/topic_result_'
	topic_result+=str(DEFAULT_DATASET)
	topic_result+='_'
	topic_result+=str(DEFAULT_TOPIC_NUMBER)
	topic_result+='_'
	topic_result+=str(DEFAULT_TRAIN_PASSES)
	topic_result+='.txt'
	topic_file=open(topic_result,'w')
	all_topics=ldamodel.show_topics(-1, 10)
	for item in all_topics:
		#print str(item)
		topic_file.write(str(item))
		topic_file.write('\n')

# get 3 type of corpus for second part test dataset
	#secondpart_texts, secondpart_corpus, secondpart_dictionary=loadtraindataset(traindataset_2,secondpart)
	with open(DEFAULT_SECOND_TEST_DATASET) as infile:
		for line in infile:
			traindataset_2.append(line)
	secondpart_corpus=loadnewdocument(traindataset_2,dictionary)

# traverse all docs in second part dataset
	count=0
	print "Generate top distribution for second part dataset"
	for old_doc_corpus in secondpart_corpus:
		#print(old_doc_corpus)
		current_distribution=ldamodel.get_document_topics(old_doc_corpus,0)
		#print "Document",count,":"
		#print(current_distribution)
		#print('\n')
		count+=1
		tempvec=[]
		for current_distribution_vec in current_distribution:
			#print(current_distribution_vec[1])
			tempvec.append(current_distribution_vec[1])
		old_document_topic_distribution.append(tempvec)	
	#print(old_document_topic_distribution)

# save result
	result_name='./result_mix_match/mix_match_result_'
	result_name+=str(DEFAULT_DATASET)
	result_name+='_'
	result_name+=str(DEFAULT_TOPIC_NUMBER)
	result_name+='_'
	result_name+=str(DEFAULT_TRAIN_PASSES)
	result_name+='.txt'
	output_file=open(result_name,'w')

# traverse all docs in first part dataset
	with open(DEFAULT_FIRST_TEST_DATASET) as firstfile:
	#with open(DEFAULT_FLICKER_FIRST) as firstfile:
		for line in firstfile:
			number_of_space=line.count(' ')
			if number_of_space<=200:
				doc_len_le_200+=1
			if number_of_space>200 and number_of_space<500:
				doc_len_200_to_500+=1
			if number_of_space>=500:
				doc_len_ge_500+=1
			print "The length of this doc is :",number_of_space
			new_document_topic_probability = []
			rank_list=dict()
			temp=[]
			temp.append(line)
			new_corpus=loadnewdocument(temp,dictionary)
# get the topic probability distribution for current doc in dataset1
			topic_distribution=ldamodel.get_document_topics(new_corpus,0)
			for new_vector in topic_distribution:
				for new_item in new_vector:
					new_document_topic_probability.append(new_item[1])

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
			temp_list=[]
			temp_list=list(rank_list)[:DEFAULT_TOP_RANK]
			temp_list_10=[]
			temp_list_10=list(rank_list)[:2]
			temp_list_20=[]
			temp_list_20=list(rank_list)[:3]

# check correct doc in top 1
			if check_list(temp_list,current_doc_index,output_file):
				#print('True')
				check_count+=1

				if number_of_space<=200:
					correct_doc_le_200_top1+=1
				if number_of_space>200 and number_of_space<500:
					correct_doc_200_to_500_top1+=1
				if number_of_space>=500:
					correct_doc_ge_500_top1+=1

# check correct doc in top 2
			if check_list_nofile(temp_list_10,current_doc_index):
				#print('True')
				check_count_10+=1

				if number_of_space<=200:
					correct_doc_le_200_top2+=1
				if number_of_space>200 and number_of_space<500:
					correct_doc_200_to_500_top2+=1
				if number_of_space>=500:
					correct_doc_ge_500_top2+=1		

# check correct doc in top 3
			if check_list_nofile(temp_list_20,current_doc_index):
				#print('True')
				check_count_20+=1

				if number_of_space<=200:
					correct_doc_le_200_top3+=1
				if number_of_space>200 and number_of_space<500:
					correct_doc_200_to_500_top3+=1
				if number_of_space>=500:
					correct_doc_ge_500_top3+=1

# check correct tag in top 1
			if check_tag(temp_list,current_doc_index):
				tag_count_1+=1

# check correct tag in top 2
			if check_tag(temp_list_10,current_doc_index):
				tag_count_2+=1

# check correct tag in top 3
			if check_tag(temp_list_20,current_doc_index):
				tag_count_3+=1

			current_doc_index+=1

	output_file.write('\n')
	print('\n')
	print('Recommendation Result:')
	output_file.write('Recommendation Result:')
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of correct doc in top 1 is:",check_count
	#print check_count
	output_file.write("The number of correct doc in top 1 is:")
	output_file.write(str(check_count))
	output_file.write('\n')
	print 100*float(check_count)/float(150),"%"
	output_file.write("The accuracy of top 1 is:")
	output_file.write(str(100*float(check_count)/float(150)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')

	print "The number of correct tag in top 1 is:",tag_count_1
	#print tag_count_1
	output_file.write("The number of correct tag in top 1 is:")
	output_file.write(str(tag_count_1))
	output_file.write('\n')
	print 100*float(tag_count_1)/float(150),"%"
	output_file.write("The accuracy of tag top 1 is:")
	output_file.write(str(100*float(tag_count_1)/float(150)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of correct doc in top 2 is:",check_count_10
	#print check_count_10
	output_file.write("The number of correct doc in top 2 is:")
	output_file.write(str(check_count_10))
	output_file.write('\n')
	print 100*float(check_count_10)/float(150),"%"
	output_file.write("The accuracy of top 2 is:")
	output_file.write(str(100*float(check_count_10)/float(150)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')

	print "The number of correct tag in top 2 is:",tag_count_2
	#print tag_count_2
	output_file.write("The number of correct tag in top 2 is:")
	output_file.write(str(tag_count_2))
	output_file.write('\n')
	print 100*float(tag_count_2)/float(150),"%"
	output_file.write("The accuracy of tag top 2 is:")
	output_file.write(str(100*float(tag_count_2)/float(150)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of correct doc in top 3 is:",check_count_20
	#print check_count_20
	output_file.write("The number of correct doc in top 3 is:")
	output_file.write(str(check_count_20))
	output_file.write('\n')
	print 100*float(check_count_20)/float(150),"%"
	output_file.write("The accuracy of top 3 is:")
	output_file.write(str(100*float(check_count_20)/float(150)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')

	print "The number of correct tag in top 3 is:",tag_count_3
	#print tag_count_3
	output_file.write("The number of correct tag in top 3 is:")
	output_file.write(str(tag_count_3))
	output_file.write('\n')
	print 100*float(tag_count_3)/float(150),"%"
	output_file.write("The accuracy of tag top 3 is:")
	output_file.write(str(100*float(tag_count_3)/float(150)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of total doc length less than 200 is:",doc_len_le_200,"\n"
	print "The number of total doc length between 200 to 500 is:",doc_len_200_to_500,"\n"
	print "The number of total doc length lager than 500 is:",doc_len_ge_500,"\n"
	#print tag_count_3
	output_file.write("The number of total doc length less than 200 is:")
	output_file.write(str(doc_len_le_200))
	output_file.write('\n')
	output_file.write("The number of total doc length between 200 to 500 is:")
	output_file.write(str(doc_len_200_to_500))
	output_file.write('\n')
	output_file.write("The number of total doc length lager than 500 is:")
	output_file.write(str(doc_len_ge_500))
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of doc length less than 200 in top 1 is:",correct_doc_le_200_top1
	print 100*float(correct_doc_le_200_top1)/float(doc_len_le_200),"%"
	print "The number of doc length between 200 to 500 in top 1 is:",correct_doc_200_to_500_top1
	print 100*float(correct_doc_200_to_500_top1)/float(doc_len_200_to_500),"%"
	print "The number of doc length lager than 500 in top 1 is:",correct_doc_ge_500_top1
	print 100*float(correct_doc_ge_500_top1)/float(doc_len_ge_500),"%"
	#print tag_count_3
	output_file.write("The number of doc length less than 200 in top 1 is:")
	output_file.write(str(correct_doc_le_200_top1))
	output_file.write('\n')
	output_file.write("The accuracy of doc length less than 200 in top 1 is:")
	output_file.write(str(100*float(correct_doc_le_200_top1)/float(doc_len_le_200)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write("The number of doc length between 200 to 500 in top 1 is:")
	output_file.write(str(correct_doc_200_to_500_top1))
	output_file.write('\n')
	output_file.write("The accuracy of doc length between 200 to 500 in top 1 is:")
	output_file.write(str(100*float(correct_doc_200_to_500_top1)/float(doc_len_200_to_500)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write("The number of doc length lager than 500 in top 1 is:")
	output_file.write(str(correct_doc_ge_500_top1))
	output_file.write('\n')
	output_file.write("The accuracy of doc length lager than 500 in top 1 is::")
	output_file.write(str(100*float(correct_doc_ge_500_top1)/float(doc_len_ge_500)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of doc length less than 200 in top 2 is:",correct_doc_le_200_top2
	print 100*float(correct_doc_le_200_top2)/float(doc_len_le_200),"%"
	print "The number of doc length between 200 to 500 in top 2 is:",correct_doc_200_to_500_top2
	print 100*float(correct_doc_200_to_500_top2)/float(doc_len_200_to_500),"%"
	print "The number of doc length lager than 500 in top 2 is:",correct_doc_ge_500_top2
	print 100*float(correct_doc_ge_500_top2)/float(doc_len_ge_500),"%"
	#print tag_count_3
	output_file.write("The number of doc length less than 200 in top 2 is:")
	output_file.write(str(correct_doc_le_200_top2))
	output_file.write('\n')
	output_file.write("The accuracy of doc length less than 200 in top 2 is:")
	output_file.write(str(100*float(correct_doc_le_200_top2)/float(doc_len_le_200)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write("The number of doc length between 200 to 500 in top 2 is:")
	output_file.write(str(correct_doc_200_to_500_top2))
	output_file.write('\n')
	output_file.write("The accuracy of doc length between 200 to 500 in top 2 is:")
	output_file.write(str(100*float(correct_doc_200_to_500_top2)/float(doc_len_200_to_500)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write("The number of doc length lager than 500 in top 2 is:")
	output_file.write(str(correct_doc_ge_500_top2))
	output_file.write('\n')
	output_file.write("The accuracy of doc length lager than 500 in top 2 is::")
	output_file.write(str(100*float(correct_doc_ge_500_top2)/float(doc_len_ge_500)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

	print "The number of doc length less than 200 in top 3 is:",correct_doc_le_200_top3
	print 100*float(correct_doc_le_200_top3)/float(doc_len_le_200),"%"
	print "The number of doc length between 200 to 500 in top 3 is:",correct_doc_200_to_500_top3
	print 100*float(correct_doc_200_to_500_top3)/float(doc_len_200_to_500),"%"
	print "The number of doc length lager than 500 in top 3 is:",correct_doc_ge_500_top3
	print 100*float(correct_doc_ge_500_top3)/float(doc_len_ge_500),"%"
	#print tag_count_3
	output_file.write("The number of doc length less than 200 in top 3 is:")
	output_file.write(str(correct_doc_le_200_top3))
	output_file.write('\n')
	output_file.write("The accuracy of doc length less than 200 in top 3 is:")
	output_file.write(str(100*float(correct_doc_le_200_top3)/float(doc_len_le_200)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write("The number of doc length between 200 to 500 in top 3 is:")
	output_file.write(str(correct_doc_200_to_500_top3))
	output_file.write('\n')
	output_file.write("The accuracy of doc length between 200 to 500 in top 3 is:")
	output_file.write(str(100*float(correct_doc_200_to_500_top3)/float(doc_len_200_to_500)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write("The number of doc length lager than 500 in top 3 is:")
	output_file.write(str(correct_doc_ge_500_top3))
	output_file.write('\n')
	output_file.write("The accuracy of doc length lager than 500 in top 3 is::")
	output_file.write(str(100*float(correct_doc_ge_500_top3)/float(doc_len_ge_500)))
	output_file.write("%")
	output_file.write('\n')
	output_file.write('\n')
	print('\n')

###############################################################################################################################################
# check top list and write into file
def check_list(toplist,doc_number,file):
	print "Check rank list for doc:",doc_number
	file.write('Rank list for doc: %s\n' % doc_number)
	for x in toplist:
		print('Doc:')
		file.write('Doc:')
		print x
		file.write(str(x))
		file.write('\n')
		if x==doc_number:
			return True
	return False

###############################################################################################################################################
# check top list not write into file
def check_list_nofile(toplist,doc_number):
	for x in toplist:
		if x==doc_number:
			return True
	return False

###############################################################################################################################################
# check top list not write into file
def check_tag(toplist,doc_number):
	if doc_number>=0 and doc_number<=49:
		for x in toplist:
			if x>=0 and x<=49:
				return True
	if doc_number>=50 and doc_number<=99:
		for x in toplist:
			if x>=50 and x<=99:
				return True
	if doc_number>=100 and doc_number<=149:
		for x in toplist:
			if x>=100 and x<=149:
				return True
	return False

###############################################################################################################################################
# main function
if __name__ == '__main__' :
	run_mix_match()
	#check_list(20)
