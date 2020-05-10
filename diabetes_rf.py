# Random Forest Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('pima-data.csv')

print(df)
print(df.shape)
#print the first 5 observation
print(df.head(5))
# print the last 5 observation
print(df.tail(5))
#check the null values
print(df.isnull().values.any())

#check the correlation

def plot_corr(df,size=11): 
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Yellow means that they are highly correlated.
                                           
    """
    corr = df.corr() # calling the correlation function on the datafrmae
    fig, ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr) # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)),corr.columns) # draw x tickmarks
    plt.yticks(range(len(corr.columns)),corr.columns) # draw y tickmarks


plot_corr(df)

#plt.show()

print(df.corr())
# delete skin columns
del df['skin']

# check the data type

print(df.head(5))
 # change from string to boolean function
print(df.head())

diabetes_map= { True :1, False:0}

df['diabetes']=df['diabetes'].map(diabetes_map)

print(df.head(5))

#checking true false
num_obs = len(df)
num_true = len(df.loc[df['diabetes'] == 1])
num_false = len(df.loc[df['diabetes'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))

# select the algorithm

from sklearn.model_selection import train_test_split

feature_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_names = ['diabetes']

X = df[feature_names].values # these are factors for the prediction
y = df[predicted_names].values # this is what we want to predict

split_test_size = 0.3

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=42)

# checking the spliting done is correctly
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

#impute the mean
# train the data  with Random Forest


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42) 
rf_model.fit(X_train,y_train.ravel())

from sklearn import metrics
#check the train dataset
rf_predict_train = rf_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train)))
print()

#check with test data

rf_predict_test = rf_model.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test)))
print()
 


print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test)))

#confusion matrix

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test,rf_predict_test)))
print("")

#performance metrics

print("Classification Report")
print("{0}".format(metrics.classification_report(y_test,rf_predict_test)))
