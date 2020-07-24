#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import pickle

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

#----------------------------------------------------

#load breast cancer data
heart_df=pd.read_csv("C:\\Users\\Ahmed\\Desktop\\prmodel\\framingham.csv")
heart_df.head()
heart_df = heart_df.drop(['education'], axis=1)

#X Data
X = heart_df.iloc[:, :-1].values
y = heart_df.iloc[:, -1].values
#----------------------------------------------------
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=44, shuffle =True)
#----------------------------------------------------
#Applying GaussianNB Model 

'''
#sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
'''
GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)

#Calculating Prediction
y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)

#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
pickle.dump(GaussianNBModel, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))



