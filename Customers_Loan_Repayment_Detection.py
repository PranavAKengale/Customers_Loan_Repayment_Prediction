import pandas as pd

data_info = pd.read_csv(r'C:\Users\prana\Downloads\Python data science\Refactored_Py_DS_ML_Bootcamp-master\22-Deep Learning\TensorFlow_FILES/DATA/lending_club_info.csv',index_col='LoanStatNew')

print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

feat_info('mort_acc')

import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv(r'C:\Users\prana\Downloads\Python data science\Refactored_Py_DS_ML_Bootcamp-master\22-Deep Learning\TensorFlow_FILES\DATA\lending_club_loan_two.csv')

df.columns

df.head(3)

df.shape

df.columns

sns.countplot(x='loan_status',data=df)

plt.figure(figsize=(12,8))
sns.distplot(df['loan_amnt'])

df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True)

#feat_info(x='installment',y='loan_amnt'),data=df

plt.figure(figsize=(12,8))
sns.scatterplot(x='installment',y='loan_amnt',data=df)

#plt.figure(figsize=(12,8))
#sns.jointplot(x='installment',y='loan_amnt',data=df,kind='kde')

plt.figure(figsize=(12,8))
sns.boxplot(x='loan_status',y='loan_amnt',data=df)

df.corr()['loan_amnt'].sort_values()

df.groupby('loan_status')['loan_amnt'].describe()

df['grade'].unique()

df['sub_grade'].unique()

feat_info('sub_grade')

plt.figure(figsize=(12,4))
subgrade_order=sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm')

plt.figure(figsize=(12,4))
subgrade_order=sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',hue='loan_status')

f_and_g=df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order=sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order=subgrade_order,palette='coolwarm',hue='loan_status')

df['loan_status'].unique()

df['loan_repaid']=df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

df[['loan_repaid','loan_status']]

df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
### Thus we can see that int_rate is highly negatively corr with load_repaid

# DATA PREPROCESSING

len(df)

df.isnull().sum()

100* df.isnull().sum()/len(df)

feat_info('emp_title')

df['emp_title'].nunique()

df['emp_title'].value_counts()

df=df.drop('emp_title',axis=1)

df['emp_length'].unique()

sorted(df['emp_length'].dropna().unique())

emp_length_order=['1 year',
    '< 1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years']

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,palette='coolwarm')

emp_co=df[df['loan_status']=='Charged Off'].groupby("emp_length").count()['loan_status']

emp_fp=df[df['loan_status']=='Fully Paid'].groupby("emp_length").count()['loan_status']

emp_len2=emp_co/(emp_co+emp_fp)*100

emp_len2.plot(kind='bar')

df=df.drop('emp_length',axis=1)

df

df.isnull().sum()

##title column is the sub category or similar like purpose column so we drop title column 

df=df.drop('title',axis=1)

df['mort_acc'].value_counts()

df.corr()['mort_acc'].sort_values()

total_acc_avg=df.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(total_acc,mort_acc):
    
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc']=df.apply(lambda x:fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)

df.isnull().sum()

df=df.dropna()

df.isnull().sum()

## Remove all the columns that are non numeric

df.select_dtypes(['object']).columns

### WE have to see this columns and check wheather they are usefull and if they are we can make 
#their dummy variables using one-hot coding


feat_info('term')

df['term'].value_counts()

df['term']=df['term'].apply(lambda term: int(term[:3]))

df['term']

df=df.drop('grade',axis=1)

dummies=pd.get_dummies(df['sub_grade'])

df=pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)

df

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

df['home_ownership'].value_counts()

df['home_ownership']=df['home_ownership'].replace(['NONE','ANY'],'OTHER')

df['home_ownership'].value_counts()

dummies=pd.get_dummies(df['home_ownership'],drop_first=True)
df=df.drop('home_ownership',axis=1)
df=pd.concat([df,dummies],axis=1)


df['address']

df['zip code']=df['address'].apply(lambda address: address[-5:])

dummies=pd.get_dummies(df['zip code'],drop_first=True)
df=df.drop('zip code',axis=1)
df=pd.concat([df,dummies],axis=1)


df=df.drop('address',axis=1)

### Actually we are purpose is to run this model that if we have to issue a loan or not 
## and in this we dont the issue date becoz it contradicts our purpose and is a data leakage


df=df.drop('issue_d',axis=1)

feat_info('earliest_cr_line')

df['earliest_cr_line']

df['earliest_cr_line']=df['earliest_cr_line'].apply(lambda earliest_cr_line: int(earliest_cr_line[-4:]))

df['earliest_cr_line']

df.columns

# Train-Test Split

from sklearn.model_selection import train_test_split

df=df.drop('loan_status',axis=1)

X=df.drop('loan_repaid',axis=1).values

X

y=df['loan_repaid'].values

df_1=df[df['loan_repaid']==1]

df_0=df[df['loan_repaid']==0]

df_1.shape

df_0.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

# Create the Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

X_train.shape

model=Sequential()

model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])




early_stop= EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=30)

model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,validation_data=(X_test,y_test),callbacks=early_stop)

# Evaluating model performance

losses=pd.DataFrame(model.history.history)

losses.plot()

from sklearn.metrics import classification_report,confusion_matrix

predictions=model.predict(X_test)

predictions.

predictions_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)] ## He step vadliye karan tensor flow 2.0 madhe
##model.predict_classes ha direct function nahiye

print(classification_report(y_test,predictions_classes))

print(confusion_matrix(y_test,predictions_classes))
