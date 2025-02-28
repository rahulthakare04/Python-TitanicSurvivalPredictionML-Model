import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# loas the data in csv
df=pandas.read_csv("titanic.csv")
# print(df)

# drop the Embarked colunm 
df=df.drop(columns=['Embarked'])

# missing value in age 
df['Age'].fillna(df['Age'].median(), inplace=True)

# missing value in fare
df['Fare'].fillna(df['Fare'].median(),inplace=True)

# drop missing valu row
df=df.dropna(subset=['Pclass','Sex','SibSp','Parch','Survived'])

df['Sex']=df['Sex'].map({'male':0,'female':1})

features=df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
labels=df['Survived']
total=0
for i in range(1,11):
    # train test model 
    feat_train,feat_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2)

    model=DecisionTreeClassifier()
    model.fit(feat_train.values,labels_train.values)

    result=model.predict(feat_test.values)
    score=accuracy_score(result,labels_test)
    total+=score
print(total)
print("Average Accuracy Score:%.2f"%((total/10)*100))