import re
import nltk
#from nltk.corpus import stopwords
from nltk.Wordcount import CountVectorizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
#from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")
nltk.download('gutenberg')

sentence_tokens = nltk.sent_tokenize(text)
word_tokens = nltk.word_tokenize(text)

pos_tags=nltk.pos_tag(tokens)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply stemming to each word in the list
st_words = [stemmer.stem(word) for word in word_list]
print(f"Original word list: {word_list}\nStemmed Word list: {st_words}\n")

# Apply lemmatization (converting words to lowercase as it works effectively on lower cases)
lem_words = [lemmatizer.lemmatize(word.lower()) for word in word_list]
print(f"Lemmatized Word list: {lem_words}")

lem_words_verb = [lemmatizer.lemmatize(word.lower(), pos='v') for word in word_list]
print(f"Lemmatized verb Word list: {lem_words_verb}")

def clean_text(t):
    t = re.sub('[^a-zA-Z]', ' ', t)
    t = t.lower()
    t = t.split()
    ps = PorterStemmer()
    t = [ps.stem(word) for word in t if word not in stopwords.words('english')]
    return " ".join(t)

df['clean_text'] = df['text'].apply(clean_text)

from nltk.corpus import stopwords

nltk.download("stopwords")

text="Hi, My name is Sara Ali. I am 19 years old. It felt nice meeting you"
word_tk=word_tokenize(text)

stop_words=set(stopwords.words("english"))

fil_words=[word for word in word_tk if word.lower() not in stop_words]
print(f"Original words: {word_tk}\nAfter removing stop words: {fil_words}")


vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
