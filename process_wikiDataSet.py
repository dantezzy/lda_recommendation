#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-14 23:40:35
# @Author  : Ziyi Zhao (zzhao37@syr.edu)
# @Version : 1.0

import os
import logging
import math
import string
import json
import sys 
import re
import os
from pprint import pprint

DEFAULT_PATH="./wikiDataSet/Arts.json"

def readjsondata(argv):

# get file name segment
	segment=argv[1].split('/')
	name_with_dot=segment[len(segment)-1]
	print("\nProcessing "+name_with_dot+" dataset")

# delete .
	purefilename=name_with_dot.split('.')
	filename=purefilename[0]

# open original file
	with open(argv[1]) as data_file:
		data=json.load(data_file) 

# write dataset into new file
	if not os.path.isdir('./pureWikiDataSet'):
		os.makedirs('./pureWikiDataSet')
	newfile=open("./pureWikiDataSet/pure_"+filename+".txt","w")

	if filename=='Computing':
		count=1
		flag_colon=':'
		flag_comma=','
		doc_str=''
		for item in data:
			if item != 0:
				print 'Doc',count
				doc_str=str(item)
                # +3 inorder to remove first : ' symbols; +3 again inorder to remove second : ' symbols;-1 inorder to remove '
				#doc_str=doc_str[doc_str.find(flag_colon)+3:]
				
				#doc_str=doc_str[doc_str.find(flag_colon)+3:]
				
				#doc_str=doc_str[doc_str.find(flag_colon)+3:]
				
				#doc_str=doc_str[:doc_str.find(flag_comma)-1]
				
				print('\n')
				newfile.write(doc_str.encode('utf-8'))
				newfile.write('\n')
				count+=1

	name_set=['Zoroastrianism','Biology','World']
	if filename in name_set:
		count=1
		flag_colon=':'
		flag_comma=','
		doc_str=''
		for item in data:
		    if item != 0:
			    print 'Doc:',count
			    doc_str=str(item)
			    # +3 inorder to remove : ' symbols; -1 inorder to remove '
			    doc_str=doc_str[doc_str.find(flag_colon)+3:doc_str.find(flag_comma)-1]
			    print(doc_str)
			    print('\n')
			    newfile.write(doc_str.encode('utf-8'))
			    newfile.write('\n')
			    count+=1
	else: #(Arts, Billionaries, Culture, Engineering, History, Sexology)
		count=1
		flag_colon=':'
		flag_comma=','
		doc_str=''
		for item in data:
			if item!=0:
				print 'Doc',count
				doc_str=str(item)
                # +3 inorder to remove first : ' symbols; +3 again inorder to remove second : ' symbols;-1 inorder to remove '
				doc_str=doc_str[doc_str.find(flag_colon)+3:]
				doc_str=doc_str[doc_str.find(flag_colon)+3:]
				doc_str=doc_str[:doc_str.find(flag_comma)-1]
				print(doc_str)
				print('\n')
				newfile.write(doc_str.encode('utf-8'))
				newfile.write('\n')
				count+=1

	print("\nNew dataset has been saved as "+"./pureWikiDataSet/pure_"+filename+".txt\n")

###############################################################################################################################################
# main function
if __name__ == '__main__' :
	readjsondata(sys.argv)