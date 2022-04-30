import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
filename ='final_data.csv'
df = pd.read_csv(filename)
print(df.shape)
print(df.head())
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
summaries = []
for sentences in df['Review']:
  inputs = tokenizer.encode("summarize: " + sentences,return_tensors='pt',max_length=512,truncation=True)
  summary_ids = model.generate(inputs, max_length = 500, min_length = 50, length_penalty = 5., num_beams = 2)
  summary = tokenizer.decode(summary_ids[0])
  summaries.append(summary)
dataframe = df
dataframe['Summaries'] = summaries
print(dataframe.shape)
dataframe.to_csv('datatest3.csv')
print("Done")