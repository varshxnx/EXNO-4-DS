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
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/3de855dd-71ba-4cbf-a4d5-715dabce1795)


```
data.isnull().sum()
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/ae769998-a9c3-4188-92c1-b622aa0a1ef0)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/4fac959a-372b-406f-b2f4-644b9a8e06f2)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/996dace7-2448-46b4-bed5-99bb9a9c6199)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/e5d1ea99-ee9b-4962-a414-d00728f86122)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/77e0d22b-0bdd-48ed-8709-b2679321f08e)

```
data2
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/d8734f54-2c6b-41e3-90be-2a1651749327)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/edce222e-22e9-43d7-9b7c-145ea12c55bb)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/56963014-1041-42d7-9285-7a7760b214c1)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/fd00fd1c-7fce-4be5-b0a8-653101f37f8a)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/e2fce328-8eda-401a-98bc-e4a788082c7e)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/356cbbb0-1648-45c8-9e64-5555f86b4d0c)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/2af0c52f-34e7-4296-bb4b-24caf51359ac)


```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/ebed5d18-129b-4a12-b0a2-f8f25303d267)

```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/2762dd1c-2007-4d5e-b779-783b270fecb1)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/5fe1fe7f-b082-4b5a-bbdf-47ddb114f35c)

```
data.shape
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/d00e97f4-266f-41c5-8ef2-cdae1c0847f7)

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
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/3856e973-eed7-4ede-a7fc-fc1a9c0196ae)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/ef993f39-b402-4dcc-af12-8bf53f55cd1a)

```
tips.time.unique()
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/1ee5f798-de96-4578-9903-d284c1d2ec5d)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/0bf40067-8b62-44e3-b820-a4e58ce720d8)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/varshxnx/EXNO-4-DS/assets/122253525/58b4458c-d12b-43ad-933f-71a258adaa78)



# RESULT:
Thus,both Feature selection and Feature scaling has been used and executed in the given dataset.

