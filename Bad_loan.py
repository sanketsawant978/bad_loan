#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")


# In[2]:


sns.set(color_codes=True)


# In[3]:


df = pd.read_csv("loan.csv")


# In[4]:


df.head()


# # Data Processing
# ***
# ## 1. Null Values:

# In[5]:


total = 163987

null_values = df.isna().sum()

percent = [round((i/total)*100,2) for i in null_values]

null = pd.DataFrame(list(zip(df.columns,null_values,percent)), columns=["Feature","Null Values","Percentage"])


# In[6]:


null.sort_values(by=["Null Values"],ascending=False,inplace=True)


# In[7]:


na = null[null["Null Values"]>0]


# In[8]:


na


# In[9]:


na["Null Values"].sum()


# In[10]:


na["Null Values"].sum()/(total*15) *100


# In[11]:


# Mean, Median, Mode
mean = []
median = []
mode = []

for i in na.Feature:
    
    mean.append(round(df[i].mean(),2))
    median.append(df[i].median())
    mode.append(df[i].mode()[0])
    
na["Mean"] = mean
na["Median"] = median
na["Mode"] = mode


# In[12]:


na


# ### Filling Missing Values

# In[13]:


df.emp_length.fillna(10.0,inplace=True)
df.revol_util.fillna(54.08,inplace=True)
df.delinq_2yrs.fillna(0.0,inplace=True)
df.total_acc.fillna(20.0,inplace=True)
df.longest_credit_length.fillna(12.0,inplace=True)
df.annual_inc.fillna(60000.0,inplace=True)


# In[14]:


df.info()


# ## 2. Categorical / Object values to Numeric

# In[15]:


labels = df.select_dtypes(include=["object"]).columns


# In[16]:


for label in labels:
    
    df[label]=LabelEncoder().fit(df[label]).transform(df[label])


# In[17]:


df.head()


# In[18]:


df.describe()


# ## 3. Feature Selection

# In[19]:


corrmat = df.corr()
corrmat.shape


# In[20]:


plt.figure(figsize=(15,8))

sns.heatmap(corrmat, annot=True)

plt.show()


# In[21]:


from sklearn.feature_selection import SelectKBest, chi2


# In[22]:


x = df.drop("bad_loan",axis=1)
y = df.bad_loan


# In[23]:


bestfeat = SelectKBest(score_func=chi2,k=10)


# In[24]:


bestfeat.fit(x,y)


# In[25]:


scores = []
for i in bestfeat.scores_:
    scores.append(round(i,2))


# In[26]:


featscores = pd.DataFrame(list(zip(x.columns,scores)),columns=["Features","Scores"])


# In[27]:


featscores.nlargest(15,"Scores")


# # Creating 1st Train Test data set

# In[28]:


x1 = x.drop(["addr_state","emp_length","delinq_2yrs"],axis=1)
y1 = y


# In[29]:


x1.columns


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3)


# #### Ada Boost

# In[31]:


AdaBoost = AdaBoostClassifier()

boostmodel = AdaBoost.fit(x_train,y_train)

y_pred = boostmodel.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_pred),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

print("\nClassification Report:\n",classification_report(y_test,y_pred))


# #### KNN

# In[32]:


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

y_predKnn = knn.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_predKnn),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predKnn))

print("\nClassification Report:\n",classification_report(y_test,y_predKnn))


# #### Decision Tree

# In[33]:


dtree = DecisionTreeClassifier(criterion="entropy",max_depth=1)

dtree.fit(x_train,y_train)

y_predtree = dtree.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_predtree),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predtree))

print("\nClassification Report:\n",classification_report(y_test,y_predtree))


# #### Random Forest

# In[34]:


rf = RandomForestClassifier(n_estimators = 20, random_state = 0)

rf.fit(x_train, y_train)

y_predrf = rf.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_predrf),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predrf))

print("\nClassification Report:\n",classification_report(y_test,y_predrf))


# #### Logistic Regression

# In[35]:


logistic = LogisticRegression()

logistic.fit(x_train, y_train)

y_predlog = logistic.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predlog),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predlog))

print("\nClassification Report:\n",classification_report(y_test,y_predlog))


# #### Navie Bayes

# In[36]:


nb = GaussianNB()

nb.fit(x_train,y_train)

y_predNB = nb.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predNB),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predNB))

print("\nClassification Report:\n",classification_report(y_test,y_predNB))


# #### Stochastic Gradient Descent

# In[37]:


sgd = SGDClassifier(loss="modified_huber", shuffle=True,random_state=0)

sgd.fit(x_train,y_train)

y_predGD = sgd.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predGD),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predGD))

print("\nClassification Report:\n",classification_report(y_test,y_predGD))


# #### Support Vector Machine SVM

# In[38]:


# svm = SVC(random_state=0)

# svm.fit(x_train,y_train)

# y_predsvm = svm.predict(x_test)

# accuracy_score(y_test, y_predsvm)


# ## 4. Handling Outliers

# In[39]:


x1.columns


# In[40]:


fig, axs = plt.subplots(4,2, figsize=(15,20))

# fig.suptitle("Box Plot",fontsize=24)

axs[0,0].boxplot(x1.loan_amnt)
axs[0,0].set_title("Loan Amount",fontsize=18)

axs[0,1].boxplot(x1.int_rate)
axs[0,1].set_title("Interest Rate",fontsize=18)

axs[1,0].boxplot(x1.annual_inc)
axs[1,0].set_title("Annual Income",fontsize=18)

axs[1,1].boxplot(x1.dti)
axs[1,1].set_title("DTI",fontsize=18)

axs[2,0].boxplot(x1.revol_util)
axs[2,0].set_title("Revolving Line",fontsize=18)

axs[2,1].boxplot(x1.total_acc)
axs[2,1].set_title("Total Account",fontsize=18)

axs[3,0].boxplot(x1.longest_credit_length)
axs[3,0].set_title("Longest Credit Length",fontsize=18)

axs[0,0].set(ylabel="Count")
axs[1,0].set(ylabel="Count")
axs[2,0].set(ylabel="Count")
axs[3,0].set(ylabel="Count")

plt.show()


# ## 4.1 Feature Engineering
# ***
#    ### 4.1.1 Loan Amount:

# In[41]:


plt.figure(figsize=(12,5))

sns.distplot(x1.loan_amnt)

plt.show()


# In[42]:


la_log = np.log(x1.loan_amnt)

plt.figure(figsize=(12,5))

sns.distplot(la_log)

plt.show()


# ### 4.1.2 Interest Rate

# In[43]:


plt.figure(figsize=(12,5))

sns.distplot(x1.int_rate)

plt.show()


# In[44]:


r = np.log(x1.int_rate)

plt.figure(figsize=(12,5))

sns.distplot(r)

plt.show()


# ### 4.1.3 Annual Income

# In[45]:


plt.figure(figsize=(12,5))

sns.distplot(x1.annual_inc)

plt.show()


# In[46]:


income = np.log(x1.annual_inc)

plt.figure(figsize=(12,5))

sns.distplot(income)

plt.show()


# ### 4.1.4 Debt to Income Ratio (DTI)

# In[47]:


plt.figure(figsize=(12,5))

sns.distplot(x1.dti)

plt.show()


# ### 4.1.5 Revolving Line

# In[48]:


plt.figure(figsize=(12,5))

sns.distplot(x1.revol_util)

plt.show()


# ### 4.1.6 Total Account

# In[49]:


plt.figure(figsize=(12,5))

sns.distplot(x1.total_acc)

plt.show()


# In[50]:


tac = np.log(x1.total_acc)

plt.figure(figsize=(12,5))

sns.distplot(tac)

plt.show()


# ### 4.1.7 Longest Credit Length

# In[51]:


plt.figure(figsize=(12,5))

sns.distplot(x1.longest_credit_length)

plt.show()


# ### New DF after transformation

# In[52]:


x2 = x1.copy()


# In[53]:


x2.columns


# In[54]:


x2.loan_amnt = la_log
x2.int_rate = r
x2.annual_inc = income


# In[55]:


fig, axs = plt.subplots(4,2, figsize=(15,20))

# fig.suptitle("Box Plot",fontsize=24)

axs[0,0].boxplot(x2.loan_amnt)
axs[0,0].set_title("Loan Amount",fontsize=18)

axs[0,1].boxplot(x2.int_rate)
axs[0,1].set_title("Interest Rate",fontsize=18)

axs[1,0].boxplot(x2.annual_inc)
axs[1,0].set_title("Annual Income",fontsize=18)

axs[1,1].boxplot(x2.dti)
axs[1,1].set_title("DTI",fontsize=18)

axs[2,0].boxplot(x2.revol_util)
axs[2,0].set_title("Revolving Line",fontsize=18)

axs[2,1].boxplot(x2.total_acc)
axs[2,1].set_title("Total Account",fontsize=18)

axs[3,0].boxplot(x2.longest_credit_length)
axs[3,0].set_title("Longest Credit Length",fontsize=18)

axs[0,0].set(ylabel="Count")
axs[1,0].set(ylabel="Count")
axs[2,0].set(ylabel="Count")
axs[3,0].set(ylabel="Count")

plt.show()


# # Creating 2nd Train Test data set

# In[56]:


x_train, x_test, y_train, y_test = train_test_split(x2, y1, test_size = 0.3)


# #### Ada Boost

# In[57]:


AdaBoost = AdaBoostClassifier()

boostmodel = AdaBoost.fit(x_train,y_train)

y_pred = boostmodel.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_pred),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

print("\nClassification Report:\n",classification_report(y_test,y_pred))


# #### Navie Bayes

# In[58]:


nb = GaussianNB()

nb.fit(x_train,y_train)

y_predNB = nb.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predNB),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predNB))

print("\nClassification Report:\n",classification_report(y_test,y_predNB))


# ## Triying Another Transformation

# In[59]:


x1.columns


# In[60]:


df1 = df.copy()


# In[61]:


df1 = df1[df1.revol_util<140]


# In[62]:


df1 = df1[df1.annual_inc<1000000]


# In[63]:


df1 = df1[df1.total_acc<100]


# In[64]:


df1 = df1[df1.longest_credit_length<60]


# In[65]:


fig, axs = plt.subplots(4,2, figsize=(15,20))

# fig.suptitle("Box Plot",fontsize=24)

axs[0,0].boxplot(df1.loan_amnt)
axs[0,0].set_title("Loan Amount",fontsize=18)

axs[0,1].boxplot(df1.int_rate)
axs[0,1].set_title("Interest Rate",fontsize=18)

axs[1,0].boxplot(df1.annual_inc)
axs[1,0].set_title("Annual Income",fontsize=18)

axs[1,1].boxplot(df1.dti)
axs[1,1].set_title("DTI",fontsize=18)

axs[2,0].boxplot(df1.revol_util)
axs[2,0].set_title("Revolving Line",fontsize=18)

axs[2,1].boxplot(df1.total_acc)
axs[2,1].set_title("Total Account",fontsize=18)

axs[3,0].boxplot(df1.longest_credit_length)
axs[3,0].set_title("Longest Credit Length",fontsize=18)

axs[0,0].set(ylabel="Count")
axs[1,0].set(ylabel="Count")
axs[2,0].set(ylabel="Count")
axs[3,0].set(ylabel="Count")

plt.show()


# In[66]:


x3 = df1.drop(["bad_loan","addr_state","emp_length","delinq_2yrs"],axis=1)
y3 = df1.bad_loan


# In[67]:


x3.columns


# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x3, y3, test_size = 0.3)


# #### Ada Boost

# In[69]:


AdaBoost = AdaBoostClassifier()

boostmodel = AdaBoost.fit(x_train,y_train)

y_pred = boostmodel.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_pred),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

print("\nClassification Report:\n",classification_report(y_test,y_pred))


# #### Navie Bayes

# In[70]:


nb = GaussianNB()

nb.fit(x_train,y_train)

y_predNB = nb.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predNB),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predNB))

print("\nClassification Report:\n",classification_report(y_test,y_predNB))


# #### KNN

# In[71]:


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

y_predKnn = knn.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_predKnn),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predKnn))

print("\nClassification Report:\n",classification_report(y_test,y_predKnn))


# #### Decision Tree

# In[72]:


dtree = DecisionTreeClassifier(criterion="entropy",max_depth=1)

dtree.fit(x_train,y_train)

y_predtree = dtree.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_predtree),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predtree))

print("\nClassification Report:\n",classification_report(y_test,y_predtree))


# #### Random Forest

# In[73]:


rf = RandomForestClassifier(n_estimators = 20, random_state = 0)

rf.fit(x_train, y_train)

y_predrf = rf.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test,y_predrf),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predrf))

print("\nClassification Report:\n",classification_report(y_test,y_predrf))


# #### Logistic Regression

# In[74]:


logistic = LogisticRegression()

logistic.fit(x_train, y_train)

y_predlog = logistic.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predlog),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predlog))

print("\nClassification Report:\n",classification_report(y_test,y_predlog))


# #### Stochastic Gradient Descent

# In[75]:


sgd = SGDClassifier(loss="modified_huber", shuffle=True,random_state=0)

sgd.fit(x_train,y_train)

y_predGD = sgd.predict(x_test)

print("Accuracy:",round(accuracy_score(y_test, y_predGD),2)*100)

print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predGD))

print("\nClassification Report:\n",classification_report(y_test,y_predGD))


# In[ ]:




