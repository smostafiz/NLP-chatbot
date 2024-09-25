import pandas as pd
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

"""Read the CSV file into a DataFrame and filter it for the company's data."""
df = pd.read_csv('twcs.csv')
company = "AppleSupport"
companyAnswers = df[df['author_id'].str.contains(company)].head(1000)

# print(companyAnswers.head(5))

# print(df[df['text'].str.contains("@" + company)].tail(30))

answerTweets = []

"""Iterate through rows in 'companyAnswers' DataFrame and collect tweets that have a response."""
for idx, t in companyAnswers[companyAnswers['author_id'].str.contains(company)].iterrows():
    if not np.isnan(t['in_response_to_tweet_id']):
        answerTweets.append(t)

questionTweets = []

# print(answerTweets[0]['in_response_to_tweet_id'])

"""Retrieve corresponding question tweets for the collected answer tweets."""
for a in answerTweets:
    question = df.loc[df['tweet_id'] == a['in_response_to_tweet_id']]
    questionTweets.append(question['text'].to_string(index=False))

"""Store the text of answer tweets in 'answerTweets'."""
for idx, t in enumerate(answerTweets):
    answerTweets[idx] = answerTweets[idx]['text']

# # Example answer and matching question
# print(answerTweets[2])
# print(questionTweets[2])
# print(len(answerTweets))
# print(len(questionTweets))

"""Preprocess the text by removing mentions and URLs."""
qListTemp = []
aListTemp = []

for t in questionTweets:
    t = re.sub('@[^\s]+', '', t)
    t = re.sub('http[^\s]+', '', t)
    qListTemp.append(t)

for t in answerTweets:
    t = re.sub('@[^\s]+', '', t)
    t = re.sub('http[^\s]+', '', t)
    aListTemp.append(t)

questionTweets = qListTemp
answerTweets = aListTemp

"""Create pairs of question and answer tweets."""
pairs = list(zip(questionTweets, answerTweets))
# print(print(pairs[66]))

"""Process and tokenize the tweets to build vocabulary."""
input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for tweet in pairs:
    input_doc, target_doc = tweet[0], tweet[1]
    input_docs.append(input_doc)

    """Remove punctuation."""
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))

    """Append START and END token to tweet."""
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    """Get each unique word from all tweets to build a vocabulary."""
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)

"""Sort and assign numeric values to tokens for one-hot encoding."""
input_tokens = sorted(list(input_tokens))
# print("INPUT TOKENS")
# print(input_tokens)
target_tokens = sorted(list(target_tokens))
# print("TARGET TOKENS")
# print(target_tokens)
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)
input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

# print("INPUT FEATURES")
# print(input_features_dict)

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

"""Initialize arrays for one-hot encoding."""
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# print("encoder input")
# print(encoder_input_data)

"""Perform one-hot encoding."""
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(target_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.
# print(pairs[:10])
