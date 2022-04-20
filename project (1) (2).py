#!/usr/bin/env python
# coding: utf-8

# ##### import pandas as pd
# dataset = pd.read_csv(r'C:\Users\janet\Downloads/train.csv',nrows=10000) 
# dataset

# In[1]:


import pandas as pd
dataset = pd.read_csv(r'C:\Users\janet\Downloads/train.csv',nrows=10000) 
dataset


# In[ ]:


dataset= dataset.drop('id' ,axis=1)

dataset


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
vectorizer = CountVectorizer(binary = True, stop_words = "english")
#X = vectorizer.fit_transform(dataset["text"])
X = vectorizer.fit_transform(dataset['text'].values.astype('U'))
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

df_tf


# In[10]:


dataset.fillna(" ")


# In[11]:


from sklearn.model_selection  import train_test_split

# partition: train/test = 70/30
train_x, test_x, train_y, test_y = train_test_split(df_tf, dataset["label"], test_size=0.3, random_state=123)

# convert numpy arrays to data frames
df_train_x = pd.DataFrame(train_x, columns=df_tf.columns)
df_test_x = pd.DataFrame(test_x, columns=df_tf.columns)
df_train_y = pd.DataFrame(train_y, columns=["target"])
df_test_y = pd.DataFrame(test_y, columns=["target"])


# In[12]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
# train model
clf = clf.fit(train_x, train_y)
# make prediction
pred_y = clf.predict(test_x)
# evaluate the prediction results

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print ("f1:" + str(f1_score(pred_y, test_y)))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y)))
print ("recall:" + str(recall_score(pred_y, test_y)))


# In[14]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()


clf = clf.fit(train_x, train_y)

# make prediction
pred_y = clf.predict(test_x)

# evaluate the prediction results
print ("f1:" + str(f1_score(pred_y, test_y)))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y)))
print ("recall:" + str(recall_score(pred_y, test_y)))


# In[15]:


x=df_tf
y=dataset['label']


# In[16]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,x,y,cv=10)
print("Accuracy: %0.2f" % scores.mean())


# In[48]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf = clf.fit(df_train_x, train_y)
pred_y = clf.predict(df_test_x)
print ("F1 using all features: %.2f" % f1_score(pred_y, test_y))
print ("accuracy: %.2f" % (accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y)))
print ("recall:" + str(recall_score(pred_y, test_y)))


# In[17]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
# train model
clf = clf.fit(train_x, train_y)
# make prediction
pred_y = clf.predict(test_x)
# evaluate the prediction results
print ("f1:" + str(f1_score(pred_y, test_y)))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y)))
print ("recall:" + str(recall_score(pred_y, test_y)))


# In[33]:


#Exploratory data analysis
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.countplot(x=dataset['label'])
real = dataset[dataset["label"]==0].values
fake = dataset[dataset["label"]==1].values
plt.bar(0,height=len(real))
plt.bar(1,height=len(fake))
plt.xticks([0,1],["Reliable News","Unreliable News"])
plt.ylabel("Counts")
plt.show()


# In[31]:





# In[19]:


from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
classifier = MultinomialNB()
from sklearn import metrics
import numpy as np
import itertools


# In[22]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix')
    
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True ')
    plt.xlabel('Predicted')


# In[23]:


classifier.fit(train_x, train_y)
pred = classifier.predict(test_x)
score = metrics.accuracy_score(test_y,pred)*100
print('Accuracy: %0.3f'% score)
cm = metrics.confusion_matrix(test_y, pred)
plot_confusion_matrix(cm, classes=['Real', 'Fake'])


# In[37]:


log = LogisticRegression(C=1e5)
log.fit(df_train_x, train_y)
count = log.predict(df_test_x)
logcount = metrics.accuracy_score(test_y,count)
print(count)
cm3 = metrics.confusion_matrix(test_y, count, labels=[0,1])
plot_confusion_matrix(cm3, classes=['TRUE','FAKE'], title ='Confusion matrix with Logistic Regression ')


# In[ ]:




