#!/usr/local/bin/python3
import csv
import pandas as pd
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

MIN_LENGTH = 50
MAX_LENGTH = 500
LENGTH_PENALTY = 5.0
NUM_BEAMS = 2

reviews = []
topics = []
with open('final_data.csv', encoding='utf-8') as f:
	csvreader = csv.reader(f)
	next(csvreader)
	for x in csvreader:
		reviews.append(x[0])
		topics.append(x[1])

summarizer = pipeline('summarization', model='facebook/bart-large-xsum')

summaries = []
for review in reviews:
	summary = summarizer(review, min_length=MIN_LENGTH, do_sample=False, truncation=True)[0]['summary_text']
	print(summary)
	print("-----------------------------")
	summaries.append(summary)

d = {'Review': reviews, 'Summary': summaries, 'Topic': topics}
df = pd.DataFrame(data=d)
df.to_csv('reviews_with_bart_summaries.csv', index=False)
"""
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')


summaries = []
for review in reviews[:10]:
	tokens = tokenizer(['summarize: ' + review], return_tensors='pt', max_length=512, truncation=True)['input_ids']
	encoded_output = model.generate(tokens, max_length=MAX_LENGTH, min_length=MIN_LENGTH, length_penalty=LENGTH_PENALTY, num_beams=NUM_BEAMS)
	summary = tokenizer.decode(encoded_output.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
	print(summary)
	print("--------------------------------------------")
	summaries.append(summary)

# print(summaries[:5])
d = {'Review': reviews, 'Summary': summaries, 'Topic': topics}
df = pd.DataFrame(data=d)
df.to_csv('reviews_with_gpt2_summaries.csv', index=False)


# encoded_output = model.generate(tokens, max_length=MAX_LENGTH, min_length=MIN_LENGTH, length_penalty=5., num_beams=2)

"""