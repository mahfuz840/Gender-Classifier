import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv('drive/MaleFemale.csv')
df_names = df
# df_names.label.replace({'positive':1,'offensive':0},inplace = True)
df_names.label.unique()
Xfeatures = df_names['name']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
cv.get_feature_names()
y = df_names.label
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test, y_test)
sample_name = ["Mahfuj Ahmed"]
vect = cv.transform(sample_name).toarray()
ans = clf.predict(vect)
if ans == 0:
    print("Female");
else:
    print("Male");
