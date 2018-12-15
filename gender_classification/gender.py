import pandas as pd 
import sklearn 
from sklearn import tree,svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae,accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
path = 'gender_data.csv'
data = pd.read_csv(path)
features = ['Height','Weight','Index']
X = data[features]
y = data.Gender
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=1)

model1 = tree.DecisionTreeClassifier(random_state=1)
model1 = model1.fit(train_X,train_y)
predictions = model1.predict(test_X)
z = accuracy_score(predictions,test_y)
print(z)

model2 = svm.SVC(gamma='auto')
model2 = model2.fit(train_X,train_y)
predictions = model2.predict(test_X)
w = accuracy_score(predictions,test_y)
print(w)

model3 = svm.LinearSVC()
model3 = model3.fit(train_X,train_y)
predictions = model3.predict(test_X)
i = accuracy_score(predictions,test_y)
print(i)

model4 = MLPClassifier()
model4 = model4.fit(train_X,train_y)
predictions = model4.predict(test_X)
p = accuracy_score(predictions,test_y)
print(p)

model5 = XGBRegressor(n_estimates=1000,learning_rate=0.01)
model5 = model5.fit(train_X,train_y)
predictions = model5.predict(test_X)
r = accuracy_score(predictions,test_y)
print(r)