import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
# -------------------------
# Custom transformer to extract numerical features
# -------------------------

df = pd.read_csv(r"C:\Users\bhara\emotion-diary-ai\data\raw\Combined Data.csv")
df.info()
class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = pd.DataFrame()
        df['word_count'] = X.apply(lambda x: len(str(x).split()))
        df['char_count'] = X.apply(lambda x: len(str(x)))
        df['avg_word_len'] = df['char_count'] / df['word_count'].replace(0,1)
        df['stopword_ratio'] = X.apply(lambda x: sum(1 for w in str(x).split() if w in ENGLISH_STOP_WORDS)/max(len(str(x).split()),1))
        df['punct_count'] = X.apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
        df['exclam_count'] = X.apply(lambda x: str(x).count('!'))
        df['question_count'] = X.apply(lambda x: str(x).count('?'))
        return df.values  # return as numpy array

# -------------------------
# Split dataset
# -------------------------
X = df['statement']
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# Pipeline
# -------------------------
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('text_stats', TextStats())
    ])),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))  # adjust model as needed
])

# -------------------------
# Train model
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
