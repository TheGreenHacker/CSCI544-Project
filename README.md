<h2>Evaluation of Summarization on Topic Extraction</h2>

This repository contains all the code required for the project by Team Tokens. The project is about assessing if summarization helps in topic classification.

<h3>Data Files:</h3>
<li> final_data.csv: final data used for summarization</li>
<li> final_summarized_BART.csv : summaries generated using BART</li>
<li> final_summarized_BERT.csv : summaries generated using BERT</li>
<li> final_summarized_T5.csv : summaries generated using T5</li>
<br>
<h3>Codes:</h3>
<li>Data_Preprocessing.py: Code to create a balanced dataset with a threshold for length of reviews. </li>
<li>bart_summarizer.py: BART summarizer</li>
<li>t5summarygeneration.py: T5 summarizer</li>
<li>bert_summarizer.py: BERT summarizer</li>
<li> Classifier.ipynb : notebook for the classification part of the project </li>
