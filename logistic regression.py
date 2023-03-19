import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
d = pd.read_csv(r"C:/Users/Administrator/Desktop/project codes/streamlit/projecttable (2).csv")

df = d.iloc[:,1:]

df.isna().sum()

a =df.describe()
print(a)

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
df['Season']= label_encoder.fit_transform(df['Season'])
  
df['Season'].unique()

df['State']= label_encoder.fit_transform(df['State'])
  
df['State'].unique()

df['Soiltype'] = label_encoder.fit_transform(df['Soiltype'])

df.dtypes
import sweetviz as sv

s = sv.analyze(df)
s.show_html()



X = df[['State','Season','AvgTemp','Rainfall','Fertilizer','Soiltype','PhValue']]
Y = df['crop']


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size = 0.2,random_state = 0)
Xtrain.shape
Xtest.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain_sc = sc.fit_transform(Xtrain)
Xtest_sc = sc.fit_transform(Xtest)
from sklearn.linear_model import LogisticRegression

# Create one-vs-rest logistic regression object
lgr = LogisticRegression(penalty='l2',C = 100,fit_intercept=True, solver="lbfgs",class_weight=None, random_state=None, max_iter=100, multi_class='auto')
# Train model
model = lgr.fit(Xtrain_sc, Ytrain)
from sklearn import metrics
predicted_values = model.predict(Xtest_sc)
x = metrics.accuracy_score(Ytest, predicted_values)
x

import numpy as np
data1 = np.array([[7,1,34,1300,130,3,6]])   
prd1 = model.predict(data1)
prd1

# View predicted probabilities

'''In one-vs-rest logistic regression (OVR) a separate model is trained for each class predicted whether an observation
 is that class or not (thus making it a binary classification problem). It assumes that each classification problem 
 (e.g. class 0 or not) is independent.'''

result =model.predict_proba(data1)
print(result)


z = ["Rice","Wheat","Bajra","Maize","Pulses","Gram","Tur","Oil seeds","Groundnut","Mustard","Sunflower","Soyabean","Cotton","Millets","Jowar","Jute","Barley","Sugarcane","Tobacco","Turmeric"]

print(result)
for r in result:
    sorted_dict = sorted({z1: r1 for r1, z1 in zip(r, z)}.items(), key=lambda x: x[1],reverse = True)
    print([c for c, _ in sorted_dict][:3])


r2 =result.tolist()


#saving the 
import pickle
with open('log_pickle.pkl','wb') as f:
    pickle.dump(model,f)
with open('log_pickle.pkl','rb') as f:
    log_pickle = pickle.load(f)
