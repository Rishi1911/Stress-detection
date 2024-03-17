#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df  = pd.read_csv("stress.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.duplicated().sum()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


df['subreddit'].value_counts()


# In[12]:


df['subreddit'].unique()


# In[13]:


plt.figure(figsize=(15,6))
sns.countplot(x= 'subreddit',data = df)
plt.xticks(rotation = 90)
plt.show()


# In[14]:


mylabels = [1,0]
plt.pie(df['label'].value_counts(),labels = mylabels,startangle = 90)
plt.show()


# In[17]:


pip install nltk


# In[15]:


import nltk


# In[16]:


import re


# In[17]:


nltk.download('stopwords')
stemmer  = nltk.SnowballStemmer('english')
from nltk.corpus import stopwords


# In[18]:


import string 


# In[19]:


stopword = set(stopwords.words('english'))


# In[20]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["text"] = df["text"].apply(clean)


# In[21]:


pip install wordcloud


# In[22]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[23]:


text = " ".join(i for i in df.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords,
background_color="black").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[24]:


df['label'] = df['label'].map({0:'no stress',1:'stress'})
df = df[['text','label']]
print(df.head())


# In[34]:


pip install sklearn 


# In[8]:


pip install scipy


# In[23]:


pip install scikit-learn


# In[25]:


import sklearn


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[27]:


x = np.array(df["text"])
y = np.array(df["label"])


# In[28]:


cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33,random_state = 42)


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB


# In[30]:


model_log = LogisticRegression()
model_log.fit(xtrain, ytrain)


# In[31]:


print("Score of the model with X-train and Y-train is : ", (model_log.score(xtrain,ytrain)))
print("Score of the model with X-test and Y-test is : ", (model_log.score(xtest,ytest)))


# In[32]:


model_dt = DecisionTreeClassifier()
model_dt.fit(xtrain,ytrain)


# In[33]:


print("Score of the model with X-train and Y-train is : ", (model_dt.score(xtrain,ytrain)))
print("Score of the model with X-test and Y-test is : ", (model_dt.score(xtest,ytest)))


# In[34]:


model_rf= RandomForestClassifier(n_estimators= 10,
criterion="entropy")
model_rf.fit(xtrain, ytrain)


# In[35]:


print("Score of the model with X-train and Y-train is : ", (model_rf.score(xtrain,ytrain)))
print("Score of the model with X-test and Y-test is : ", (model_rf.score(xtest,ytest)))


# In[36]:


model = BernoulliNB()
model.fit(xtrain, ytrain)


# In[37]:


print("Score of the model with X-train and Y-train is : ", (model.score(xtrain,ytrain)))
print("Score of the model with X-test and Y-test is : ", (model.score(xtest,ytest)))


# In[40]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model_dt.predict(data)
print(output)


# In[41]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model_dt.predict(data)
print(output)


# In[ ]:




