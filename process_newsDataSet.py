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

DEFAULT_PATH="./newsDataSet/business.json"

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
	if not os.path.isdir('./pureNewsDataSet'):
		os.makedirs('./pureNewsDataSet')
	newfile=open("./pureNewsDataSet/pure_"+filename+".txt","w")

	count=1
	for item in data:
		if item != 0:
			print 'Doc:',count
			print(item[4])
			print('\n')
			newfile.write(item[4].encode('utf-8'))
			newfile.write("\n")
			count+=1

	print("\nNew dataset has been saved as "+"./pureNewsDataSet/pure_"+filename+".txt\n")

###############################################################################################################################################
# main function
if __name__ == '__main__' :
	readjsondata(sys.argv)