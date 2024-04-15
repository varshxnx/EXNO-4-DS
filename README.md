# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1: Read the given Data.
STEP 2: Clean the Data Set using Data Cleaning Process.
STEP 3: Apply Feature Scaling for the feature in the data set.
STEP 4: Apply Feature Selection for the feature in the data set.
STEP 5: Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1. Filter Method
2. Wrapper Method
3. Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/ff7cccdb-bd60-45dd-9dc6-47b49a563538)

```
data.isnull().sum()
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/59b6ad1a-bf12-48c2-a294-5aac15886114)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/33a8312b-2d89-4abd-bb0d-35cdfa2b7b44)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/84785d2a-5779-49fa-b399-8edbd5892b9e)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/dcbc0758-336f-4d51-8fe8-e685afb2ed93)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/0f52b3b1-2144-4e94-ba27-03ca586215f0)

```
data2
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/ffb9c16e-40c4-4439-a29f-b201581a12c8)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/de84fef7-7332-46d8-ab50-471d0169a901)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/ed5ee91b-8bf1-4e8c-a8cf-3d9d96a5589d)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/5ddd2e17-6819-4f8b-acba-90cf4897797a)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/ba4ffd91-efc5-4987-8486-49b620e17f41)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/e0f0561b-97cf-4176-bbb5-19614b52c408)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/4a77fca3-3948-4c11-bdc0-514a61afceab)

```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/3f22074f-9d4d-4758-962c-57f12b70146b)
```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/e31a4e64-7fca-4531-a188-48a5ff07266e)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/d7291f25-f68a-4c7b-b781-a745058b2770)
```
data.shape
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/bcaaa675-3cb4-477f-83b5-5d4fdea4d996)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/9263244e-6532-4827-8413-9a0633efbf7d)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/f6ad3642-5ec2-4c93-88c0-0019e3127b90)
```
tips.time.unique()
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/f4b72a8c-b35a-40df-8649-9123983f7704)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/9cd46b28-eda2-44b7-82c2-277dfac3bcc5)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/Vanitha-SM/EXNO-4-DS/assets/119557985/2fff30ed-ac94-411c-9256-51bf5928d9b9)


# RESULT:
Thus,both Feature selection and Feature scaling has been used and executed in the given dataset.

