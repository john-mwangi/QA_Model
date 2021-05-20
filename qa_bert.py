from numpy.lib.function_base import append
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import pandas as pd

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big… it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

input_ids = tokenizer.encode(question, answer_text)
print(f'The input has a total of {len(input_ids)} tokens')

tokens = tokenizer.convert_ids_to_tokens(input_ids)

for token,id in zip(tokens, input_ids):
    if id == tokenizer.sep_token_id:
        print('')

    print('{:<12} {:>6,}'.format(token,id))
    #print(f'{token} {id}')

sep_index = input_ids.index(tokenizer.sep_token_id)

#Segment A tokens (the question)
num_seg_a = sep_index + 1

#Segment B tokens (the answer)
num_seg_b = len(input_ids) - num_seg_a

#Construct a list of 0 & 1s
segment_ids = [0]*num_seg_a + [1]*num_seg_b

#Check if each token has a segment id
assert len(segment_ids)==len(input_ids)

#Transformers inputs are of type tensor
outputs = model(torch.tensor([input_ids]), token_type_ids = torch.tensor([segment_ids]))

#outputs is an object of class "QuestionAnsweringModelOutput" with start_logits & end_logits
start_scores = outputs.start_logits
end_scores = outputs.end_logits

#The answer is represented by the most probable start and end of the answer
#These represent the positions of the tokens containing the answer
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

tokens[answer_start:answer_end + 1]

answer = "".join(tokens[answer_start:answer_end+1])

#Format the answer
answer = tokens[answer_start]

for i in range(answer_start+1, answer_end+1):
    if tokens[i][0:2]=="##": #remove these characters
        answer += tokens[i][2:]
    else:
        answer += " "+tokens[i] #add space before token

def answer_question(question, answer_text):
    
    input_ids = tokenizer.encode(question, answer_text)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    outputs = model(torch.tensor([input_ids]), token_type_ids = torch.tensor([segment_ids]))

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    answer = tokens[answer_start]

    for i in range(answer_start+1, answer_end+1):
        if tokens[i][0:2]=="##":
            answer += tokens[i][2:]
        else:
            answer += " "+tokens[i]

    print(f'Answer: {answer}')


question="What does the 'B' in BERT stand for?"
question="What are some example applications of BERT?"
answer_text="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."


answer_question(question=question, answer_text=answer_text)

quarter_1 = pd.read_excel("./Amazon/Q1.xlsx", header=None, names=['quarter'])

quarter_1.head

quarter_1_str = quarter_1.quarter.astype(str)

quarter_1_str = ''.join(quarter_1_str)

len(quarter_1_str)

with open('Amazon/Q1.txt', mode='r', encoding="utf8") as f:
    quarter_1_text = f.readlines()

quarter_1_text

question = "What was the operating cash flow?"

answer_text = "AMAZON.COM ANNOUNCES FIRST QUARTER RESULTS SEATTLE—(BUSINESS WIRE) April 30, 2020—Amazon.com, Inc. (NASDAQ: AMZN) today announced financial results for its first quarter ended March 31, 2020. Operating cash flow increased 16% to $39.7 billion for the trailing twelve months, compared with $34.4 billion for the trailing twelve months ended March 31, 2019. Free cash flow increased to $24.3 billion for the trailing twelve months, compared with $23.0 billion for the trailing twelve months ended March 31, 2019. Free cash flow less principal repayments of finance leases and financing obligations decreased to $14.3 billion for the trailing twelve months, compared with $15.1 billion for the trailing twelve months ended March 31, 2019. Free cash flow less equipment finance leases and principal repayments of all other finance leases and financing obligations decreased to $11.7 billion for the trailing twelve months, compared with $11.8 billion for the trailing twelve months ended March 31, 2019. Common shares outstanding plus shares underlying stock-based awards totaled 513 million on March 31, 2020, compared with 507 million one year ago. Net sales increased 26% to $75.5 billion in the first quarter, compared with $59.7 billion in first quarter 2019. Excluding the $387 million unfavorable impact from year-over-year changes in foreign exchange rates throughout the quarter, net sales increased 27% compared with first quarter 2019. Operating income decreased to $4.0 billion in the first quarter, compared with operating income of $4.4 billion in first quarter 2019. Net income decreased to $2.5 billion in the first quarter, or $5.01 per diluted share, compared with net income of $3.6 billion, or $7.09 per diluted share, in first quarter 2019. Net income decreased to $2.5 billion in the first quarter, or $5.01 per diluted share, compared with net income of $3.6 billion, or $7.09 per diluted share, in first quarter 2019. From online shopping to AWS to Prime Video and Fire TV, the current crisis is demonstrating the adaptability and durability. of Amazon’s business as never before, but it’s also the hardest time we’ve ever faced said Jeff Bezos, Amazon founder and CEO. We are inspired by all the essential workers we see doing their jobs — nurses and doctors, grocery store cashiers, police."

answer_question(question=question, answer_text=answer_text)

len(answer_text)

len(tokenizer.encode(question, answer_text))


