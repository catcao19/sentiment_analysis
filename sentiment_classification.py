#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('reviews.csv', delimiter='\t')


# In[2]:


data


# In[3]:


def binned_ratings(n):
    if (n > 3):
        return 2
    elif (n < 3):
        return 0
    else: 
        return 1


# In[4]:


data['RatingValue'] = data['RatingValue'].apply(lambda x: binned_ratings(x))


# In[5]:


data['RatingValue'].value_counts()


# In[6]:


df = data.groupby('RatingValue')
df_1 = df.apply(lambda x: x.sample(n = 158, random_state = 20)).reset_index(drop = True)


# In[7]:


df2 = pd.concat([df_1['RatingValue'], df_1['Review']], axis = 1)
df2.columns = ['Sentiment', 'Review']
df2


# In[8]:


df2['Sentiment'].value_counts()


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2['Review'], df2['Sentiment'], test_size = 0.2, random_state = 20, shuffle = True)


# In[10]:


#X_train,y_train.to_csv('train.csv', sep='\t', index=False)
#X_test,y_test.to_csv('valid.csv', sep='\t', index=False)
train_df = pd.DataFrame(data = [y_train, X_train]).T
train_df.to_csv('train.csv')


# In[11]:


train_df = pd.DataFrame(data = [y_test, X_test]).T
train_df.to_csv('valid.csv')


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# In[13]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[14]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])


# In[16]:


train = pd.read_csv('train.csv')


# In[17]:


text_clf.fit(train['Review'], train['Sentiment'])


# In[18]:


valid = pd.read_csv('valid.csv')


# In[19]:


import numpy as np
predicted = text_clf.predict(valid['Review'])
accuracy = np.mean(predicted == valid['Sentiment'])
print('accuracy on the test set: {:.3f}'.format(accuracy))


# In[20]:


from sklearn.metrics import f1_score
f1_score = f1_score(valid['Sentiment'], predicted, average='macro')
print('f1 score on the test set: {:.3f}'.format(f1_score))


# In[26]:


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(valid['Sentiment'], predicted)
confusion_df = pd.DataFrame(confusion)
confusion_df.columns = ['negative', 'neutral', 'positive']
confusion_df.index = ['negative', 'neutral', 'positive']
print('Confusion_matrix:')
print(confusion_df)


# In[ ]:


#train['Sentiment'].value_counts() / train.shape[0]


# In[ ]:




