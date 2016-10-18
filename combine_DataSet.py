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

file_list_test=["pureWikiDataSet/pure_Zoroastrianism.txt"]

file_list_1=["pureWikiDataSet/pure_Arts.txt",
           "pureWikiDataSet/pure_Billionaires.txt",
           "pureWikiDataSet/pure_Biology.txt",
           "pureWikiDataSet/pure_Culture.txt",
           "pureWikiDataSet/pure_Engineering.txt",
           "pureWikiDataSet/pure_History.txt",
           "pureWikiDataSet/pure_Sexology.txt",
           "pureWikiDataSet/pure_Society.txt",
           "pureWikiDataSet/pure_Telecommunications.txt",
           "pureWikiDataSet/pure_World.txt",
           "pureWikiDataSet/pure_Zoroastrianism.txt"]

file_list_2=["pureNewsDataSet/pure_business.txt",
             "pureNewsDataSet/pure_politics.txt",
             "pureNewsDataSet/pure_scitech.txt",
             "pureNewsDataSet/pure_sport.txt"]

###############################################################################################################################################
# combine multiple dataset
def combine_dataset():
	with open('wiki_dataset.txt', 'w') as outfile:
		for fname in file_list_test:
			print 'Process :',fname
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)

###############################################################################################################################################
# main function
if __name__ == '__main__' :
	combine_dataset()