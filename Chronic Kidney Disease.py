import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

Read_data = pd.read_csv('kidney_disease.csv')

Read_data.drop('id', axis =1, inplace=True)

Read_data.describe()
Read_data.info()

Read_data.columns = ['Age', 'Blood_Pressure', 'Specific_Gravity', 
                     'Albumin', 'Sugar', 'Red_Blood_cells', 'Pus_Cell',
                       'Pus_Cell_Clumps', 'Bacteria', 'Blood_Glucose_Random',
                       'Blood_Ures','Serum_Creatinine', 'Sodium', 'Potsassium',
                       'Haemoglobin','Packed_Cell_Volume','White_Blood_Cell_Count',
                       'Red_Blood_Cell_Count','Hypertension','Diabetes_Mellitus', 
                       'Coronary_Artery_Disease','Appetite','Peda_Edema', 'Anemia',
                       'Classification']

text_column = ['Packed_Cell_Volume','White_Blood_Cell_Count','Red_Blood_Cell_Count']
for i in text_column:
    print(f'{i} : {Read_data[i].dtype}')


def numericConversion(Read_data, Column):
    Read_data[Column] = pd.to_numeric(Read_data[Column], errors ='coerce')

for Column in text_column:
    numericConversion(Read_data, Column)
    print(f'{Column} : {Read_data[Column].dtype}')


missingValues = Read_data.isnull().sum()
missingValues[missingValues>0].sort_values(ascending=False)



def mean_value(Read_data, Column):
    mean_Value= Read_data[Column].mean()
    Read_data[Column].fillna(value= mean_Value, inplace=True)

def mode_value(Read_data, Column):
    mode_Value=Read_data[Column].mode()[0]
    Read_data[Column]= Read_data[Column].fillna(mode_Value)


num=[col for col in Read_data.columns if Read_data[col].dtype!='object']

for i in num:
    mean_value(Read_data, i)

obj= [col for col in Read_data.columns if Read_data[col].dtype=='object']

for i in obj:
    mode_value(Read_data, i)



print(f'Diabetes_Mellitus : {Read_data['Diabetes_Mellitus'].unique()}')
print(f'Coronary_Artery_Disease	 : {Read_data['Coronary_Artery_Disease'].unique()}')
print(f'Classification : {Read_data['Classification'].unique()}')



Read_data['Diabetes_Mellitus']=Read_data['Diabetes_Mellitus'].replace(to_replace={' yes':'yes', '\tno':'no','\tyes':'yes'})
Read_data['Coronary_Artery_Disease']=Read_data['Coronary_Artery_Disease'].replace(to_replace={'\tno':'no'})
Read_data['Classification']=Read_data['Classification'].replace(to_replace={'ckd\t':'ckd','notckd':'not ckd'})


Read_data['Red_Blood_cells'] = Read_data['Red_Blood_cells'].map({'normal':1,'abnormal':0})
Read_data['Pus_Cell'] = Read_data['Pus_Cell'].map({'normal':1,'abnormal':0})
Read_data['Pus_Cell_Clumps'] = Read_data['Pus_Cell_Clumps'].map({'present':1,'notpresent':0})
Read_data['Bacteria'] = Read_data['Bacteria'].map({'present':1,'notpresent':0})
Read_data['Hypertension'] = Read_data['Hypertension'].map({'yes':1,'no':0})
Read_data['Diabetes_Mellitus'] = Read_data['Diabetes_Mellitus'].map({'yes':1,'no':0})
Read_data['Coronary_Artery_Disease']  =Read_data['Coronary_Artery_Disease'].map({'yes':1,'no':0})
Read_data['Appetite'] = Read_data['Appetite'].map({'good':1,'poor':0})
Read_data['Peda_Edema'] = Read_data['Peda_Edema'].map({'yes':1,'no':0})
Read_data['Anemia'] = Read_data['Anemia'].map({'yes':1,'no':0})
Read_data['Classification'] = Read_data['Classification'].map({'ckd':1,'not ckd':0})


plt.figure(figsize=(15,8))
sns.heatmap(Read_data.corr(), annot=True, linewidths=0.5)
plt.show()


TargetCorr = Read_data.corr()['Classification'].abs().sort_values(ascending=False)

x = Read_data.drop('Classification', axis=1)
y = Read_data['Classification']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=25)

print(x_train.shape)
print(x_test.shape)

dct= DecisionTreeClassifier()
dct.fit(x_train, y_train)

model =[]
model.append(('Navie Bayes' , GaussianNB()))
model.append(('KNN', KNeighborsClassifier(n_neighbors=8)))
model.append(('RandomForestClassifier', RandomForestClassifier()))
model.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
model.append(('SVM', SVC(kernel='linear')))

for name, models in model:
    print(name, models)
    print()
    models.fit(x_train, y_train)
    y_predt= models.predict(x_test)
    print(confusion_matrix(y_test,y_predt))
    print('\n')
    print('accuracy_score', accuracy_score(y_test,y_predt))
    print('\n')
    print('precision_score', precision_score(y_test,y_predt))
    print('\n')
    print('recall_score',recall_score(y_test,y_predt))
    print('\n')
    print('f1_score', f1_score(y_test,y_predt))
    print('\n')