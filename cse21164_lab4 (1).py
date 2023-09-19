#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import math

data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)
print(df)
def entropy(attribute):
    entropy = 0
    values = df[attribute].unique()
    for v in values:
        p = len(df[df[attribute] == v]) / len(df)
        entropy += -p * math.log2(p)
    return entropy


income = entropy('income')
credit_rating = entropy('credit_rating')
buys_computer = entropy('buys_computer')
student = entropy('student')
age = entropy('age')

def information_gain(attribute):
    values = df[attribute].unique()
    information_gain = buys_computer_entropy
    for v in values:
        subset = df[df[attribute] == v]
        p = len(subset) / len(df)
        information_gain -= p * entropy('buys_computer')
    return information_gain


age_ig = information_gain('age')
income_ig = information_gain('income')
student_ig = information_gain('student')
credit_rating_ig = information_gain('credit_rating')

print('Income Entropy:', income)
print('Credit Rating Entropy:', credit_rating)
print('Buys Computer Entropy:', buys_computer)
print('Student Entropy:', student)
print('Age Entropy:', age)

print('Information_Gain_Age:', age_ig)
print('Information_Gain_Income:', income_ig)
print('Information_Gain_Student:', student_ig)
print('Information_Gain_Credit Rating:', credit_rating_ig)

 

n = max(age_ig, income_ig, student_ig, credit_rating_ig)
if n == age_ig:
    print('first feature selected for decision tree construction is Age.')
elif n == income_ig:
    print('first feature selected for decision tree construction is Income.')
elif n == student_ig:
    print('first feature selected for decision tree construction is Student.')
else:
    print('first feature selected for decision tree construction is Credit Rating.')


# In[13]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = {

    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],

    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],

    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],

    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],

    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

}

df = pd.DataFrame(data)
df_encoded = df.apply(lambda col: pd.factorize(col)[0])
Train_X = df_encoded.drop(columns=['buys_computer'])
Train_y = df_encoded['buys_computer']
model = DecisionTreeClassifier()
model.fit(Train_X, Train_y)
accuracy_for_training = model.score(Train_X, Train_y)

print(f"Training Set Accuracy: {accuracy_for_training}")

t_depth = model.get_depth()

print(f"Tree Depth: {t_depth}")


# In[ ]:




