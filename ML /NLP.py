import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")
nltk.download('gutenberg')

sentence_tokens = sent_tokenize(text)
word_tokens = word_tokenize(text)


def clean_text(t):
    t = re.sub('[^a-zA-Z]', ' ', t)
    t = t.lower()
    t = t.split()
    ps = PorterStemmer()
    t = [ps.stem(word) for word in t if word not in stopwords.words('english')]
    return " ".join(t)

df['clean_text'] = df['text'].apply(clean_text)
