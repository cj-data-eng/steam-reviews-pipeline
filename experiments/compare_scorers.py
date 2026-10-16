import glob

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

files = glob.glob("data/gold/reviews_enriched/**/*.parquet", recursive=True)
df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
df = df.sort_values("created_at").reset_index(drop=True)
y = df["voted_up"].astype(int)

cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

print(f"train {len(train)} rows: {train.created_at.min()} -> {train.created_at.max()}")
print(f"test  {len(test)} rows: {test.created_at.min()} -> {test.created_at.max()}")
print(f"share recommended: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
print(f"always-positive accuracy on test: {y_test.mean():.3f}")

print(f"VADER AUC on test: {roc_auc_score(y_test, test.sentiment_score):.3f}")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
x_train = vec.fit_transform(train.review_text)
x_test = vec.transform(test.review_text)
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train)
proba = clf.predict_proba(x_test)[:, 1]
print(f"TF-IDF + LogReg AUC on test: {roc_auc_score(y_test, proba):.3f}")
print(f"TF-IDF + LogReg accuracy on test: {((proba >= 0.5) == y_test).mean():.3f}")

names = vec.get_feature_names_out()
order = clf.coef_[0].argsort()
print("most negative terms:", [names[i] for i in order[:15]])
print("most positive terms:", [names[i] for i in order[-15:]])