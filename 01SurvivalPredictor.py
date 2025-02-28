import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# loas the data in csv
df=pandas.read_csv("titanic.csv")
# print(df)

# drop the Embarked colunm 
df=df.drop(columns=['Embarked'])

# missing value in age 
df['Age'].fillna(df['Age'].median())

# missing value in fare
df['Fare'].fillna(df['Fare'].median())

# drop missing valu row
df=df.dropna(subset=['Pclass','Sex','SibSp','Parch','Survived'])

df['Sex']=df['Sex'].map({'male':0,'female':1})

features=df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
labels=df['Survived']

model=DecisionTreeClassifier()
model.fit(features.values,labels.values)


result=model.predict([[2,1,36,0,1,22254],[1,0,25,1,1,40000]])
print(result)