#!/usr/local/bin/python3
import csv
import pandas as pd
from transformers import pipeline
from summarizer import Summarizer

MIN_LENGTH = 70
MAX_LENGTH = 120

reviews = []
topics = []
with open('final_data.csv', encoding='utf-8') as f:
	csvreader = csv.reader(f)
	next(csvreader)
	for x in csvreader:
		reviews.append(x[0])
		topics.append(x[1])


summarizer = Summarizer()
summaries = []
for review in reviews:
	summary = summarizer(review, min_length=MIN_LENGTH, max_length=MAX_LENGTH)
	print(summary)
	print("-----------------------------")
	summaries.append(summary)

d = {'Review': reviews, 'Summary': summaries, 'Topic': topics}
df = pd.DataFrame(data=d)
df.to_csv('reviews_with_bert_summaries.csv', index=False)