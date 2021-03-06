"""
Question answering model using the BERT transformer.
"""

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

question = "What does the 'B' in BERT stand for?"
answer_text = "We introduce a new language representation model called BERT, \
    which stands for Bidirectional Encoder Representations from Transformers. \
    Unlike recent language representation models (Peters et al., 2018a; \
    Radford et al., 2018), BERT is designed to pretrain deep bidirectional \
    representations from unlabeled text by jointly conditioning on both left \
    and right context in all layers. As a result, the pre-trained BERT model \
    can be finetuned with just one additional output layer to create state-of-the-art \
    models for a wide range of tasks, such as question answering and language \
    inference, without substantial taskspecific architecture modifications. \
    BERT is conceptually simple and empirically powerful. It obtains new \
    state-of-the-art results on eleven natural language processing tasks, \
    including pushing the GLUE score to 80.5% (7.7% point absolute improvement), \
    MultiNLI accuracy to 86.7% (4.6% absolute improvement), \
    SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) \
    and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."

encodings = tokenizer(text=question, text_pair=answer_text, truncation=True, padding=True)

#This contains the inputs required to the model.
encodings.keys()
"""
tokens: sentence broken down to individual words
input_ids: the index of the token
segment: flags the different parts of a sentence. They are indicated by token_type_ids
scores: these are generated by the model itself and indicate where the answer starts and ends
"""
input_ids = encodings.get("input_ids")
segment_embeddings = encodings.get("token_type_ids")
input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

outputs = model(input_ids=torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_embeddings]))

start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

answer = "".join(input_tokens[start_index:end_index + 1])

# BERT returns stemmed results
cleaned_answer = ""
for i in range(start_index, end_index + 1):
    if input_tokens[i][:2] == "##":
        cleaned_answer += input_tokens[i][2:]
    else:
        cleaned_answer += " " + input_tokens[i]

def question_answer(question, answer_text):
    encodings = tokenizer(text=question, text_pair=answer_text, truncation=True, padding=True)

    input_ids = encodings.get("input_ids")
    segment_embeddings = encodings.get("token_type_ids")
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    outputs = model(input_ids=torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_embeddings]))
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    cleaned_answer = ""
    for i in range(start_index, end_index + 1):
        if input_tokens[i][:2] == "##":
            cleaned_answer += input_tokens[i][2:]
        else:
            cleaned_answer += " " + input_tokens[i]
    
    print(f"Answer: {cleaned_answer}")

# Test
question_2 = "What was the operating cash flow?"

answer_text_2 = "AMAZON.COM ANNOUNCES FIRST QUARTER RESULTS \
SEATTLE???(BUSINESS WIRE) April 30, 2020???Amazon.com, Inc. (NASDAQ: AMZN) today announced financial results \
for its first quarter ended March 31, 2020. \
Operating cash flow increased 16% to $39.7 billion for the trailing twelve months, compared with $34.4 billion for \
the trailing twelve months ended March 31, 2019. \
Free cash flow increased to $24.3 billion for the trailing twelve months, compared with $23.0 billion for the trailing \
twelve months ended March 31, 2019. \
Free cash flow less principal repayments of finance leases and financing obligations decreased to $14.3 billion for \
the trailing twelve months, compared with $15.1 billion for the trailing twelve months ended March 31, 2019. \
Free cash flow less equipment finance leases and principal repayments of all other finance leases and financing \
obligations decreased to $11.7 billion for the trailing twelve months, compared with $11.8 billion for the trailing \
twelve months ended March 31, 2019. \
Common shares outstanding plus shares underlying stock-based awards totaled 513 million on March 31, 2020, \
compared with 507 million one year ago. \
Net sales increased 26% to $75.5 billion in the first quarter, compared with $59.7 billion in first quarter 2019. \
Excluding the $387 million unfavorable impact from year-over-year changes in foreign exchange rates throughout the \
quarter, net sales increased 27% compared with first quarter 2019. \
Operating income decreased to $4.0 billion in the first quarter, compared with operating income of $4.4 billion in \
first quarter 2019. \
Net income decreased to $2.5 billion in the first quarter, or $5.01 per diluted share, compared with net income of $3.6 \
billion, or $7.09 per diluted share, in first quarter 2019. \
Net income decreased to $2.5 billion in the first quarter, or $5.01 per diluted share, compared with net income of $3.6 \
billion, or $7.09 per diluted share, in first quarter 2019. \
From online shopping to AWS to Prime Video and Fire TV, the current crisis is demonstrating the adaptability and durability. \
of Amazon???s business as never before, but it???s also the hardest time we???ve ever faced said Jeff Bezos, Amazon founder and \
CEO. We are inspired by all the essential workers we see doing their jobs ??? nurses and doctors, grocery store cashiers, police."

question_answer(question=question_2, answer_text=answer_text_2)

# CHECKS
# Length of input tokens
len(tokenizer.encode(text=question_2, text_pair=answer_text_2))

# Max length of input tokens
tokenizer.model_max_length
