#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:51:17 2017

@author: denispetruchik
"""

import pandas as pd
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re


df = pd.read_csv('/Users/denispetruchik/Downloads/train.csv')
y = df['Survived']
X = df.drop({'PassengerId','Survived'}, axis = 1)

import matplotlib.pyplot as plt
import seaborn as sns

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def plot_correlation_map( df, y ):
    pl = df.copy()
    pl['Survived'] = y
    corr = pl.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print test_scores_mean[test_scores_mean.size - 1]
    plt.axhline(linewidth=1, color='b', y=0.8)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

from sklearn.ensemble import RandomForestRegressor
#predicting missing values in age using Random Forest
def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp', 'IsAlone', 'Large_Family', 'BadTicket', 'Title','Pclass','Family','Deck']]
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df

# Convert categorical features using one-hot encoding.
def onehot(onehot_df, df, column_name, fill_na=None, drop_name=None):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)

    dummies = pd.get_dummies(onehot_df[column_name], prefix="_" + column_name)
    
    # Dropping one of the columns actually made the results slightly worse.
    # if drop_name is not None:
    #     dummies.drop(["_" + column_name + "_" + drop_name], axis=1, inplace=True)

    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)
    return onehot_df


def processX(X):

    X['Title'] = X['Name'].apply(get_title)
    X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt','Dr', 'Col','Don', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms',   'Miss')
    X['Title'] = X['Title'].replace('Mme',  'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    X['Title'] = X['Title'].map(title_mapping)
    X['Title'] = X['Title'].fillna(6)

    X['Embarked'] = X['Embarked'].fillna('S')
    X['Embarked'] = X['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    X['BadTicket'] = X['Ticket'].str[0].isin(['3','4','5','6','7','8','A','L','W'])

    X['Deck'] = X['Cabin'].str[0]
    X['Deck'] = X['Deck'].fillna(value='U')
    labelEnc=LabelEncoder() 
    X['Deck'] = labelEnc.fit_transform(X['Deck'])


    X['Has_Cabin'] = X['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    X['Sex'] = X['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    X['Family'] = X['SibSp'] + X['Parch']
    X['Large_Family'] = (X['SibSp']>2) | (X['Parch']>3)
    X['IsAlone']  = (X['SibSp'] + X['Parch']) == 0

    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    X.loc[ X['Fare'] <= 7.91,                            'FareCl'] = 0
    X.loc[(X['Fare'] >  7.91)   & (X['Fare'] <= 14.454), 'FareCl'] = 1
    X.loc[(X['Fare'] >  14.454) & (X['Fare'] <= 31),     'FareCl'] = 2
    X.loc[ X['Fare'] >  31,                              'FareCl'] = 3

    fill_missing_age(X)
    X['Child'] =  X['Age'] <=10
#    X['Young'] = (X['Age'] <=30) & (X['Age'] > 10) # | (X['Title'].isin([2,3]))
    X['Young'] = (X['Age'] <=30) | (X['Title'].isin([2,3]))
    X['Age'] = X['Age'].astype(int)

    
    X['Fare'] = X['Fare'].astype(int)
    X['FareCl'] = X['FareCl'].astype(int)
    X['SharedTicket'] = np.where(X.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
    from sklearn import preprocessing
    
    std_scale = preprocessing.StandardScaler().fit(X[['Age', 'Fare']])
    X[['Age', 'Fare']] = std_scale.transform(X[['Age', 'Fare']])    
#    X = onehot(X, X, 'Deck')    
    X = onehot(X, X, 'FareCl')
    X = onehot(X, X, 'Title')
    X.drop({'Ticket', 'Family','Cabin','SibSp', 'Parch', 'Name'}, axis = 1, inplace = True)

#    X_prev = X.copy()
    
    
#    X,e = encode_with_OneHotEncoder_and_delete_column(X,'Person')
#    X,e = encode_with_OneHotEncoder_and_delete_column(X,'FareCl')
#    X,e = encode_with_OneHotEncoder_and_delete_column(X,'Sex')
#    X,e = encode_with_OneHotEncoder_and_delete_column(X,'Embarked')
#    X,e = encode_with_OneHotEncoder_and_delete_column(X,'Title')
    return X

X = processX(X)
#plot_correlation_map(X, y)
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
from sklearn.model_selection  import cross_val_score
from sklearn.cross_validation import KFold
kf = KFold(y.count(), n_folds=5, shuffle=True, random_state=1)

# fit model no training data
model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=400, nthread=-1,
       objective='binary:logistic', reg_alpha=3.0, reg_lambda=4.0,
       scale_pos_weight=1, seed=42, silent=True, subsample=1)

plot_learning_curve(model, 'XGBClassifier', X, y, cv=4);
model = model.fit(X,y)
print model.booster().get_fscore()

from sklearn.ensemble import RandomForestClassifier

def determine_forest_quality(n_estimators):
    random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=12,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=42,
            verbose=0, warm_start=False)
    return  cross_val_score(random_forest, X, y, scoring='precision', cv=10).mean()


#for k in range(100,501,100): print (k, determine_forest_quality(k))

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=12,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=True, random_state=13,
            verbose=0, warm_start=False)

#plot_learning_curve(random_forest, 'Random Forest', X, y, cv=4);
print cross_val_score(random_forest, X, y, scoring='precision', cv=4).mean()
random_forest = random_forest.fit(X, y)



from catboost import CatBoostClassifier
categorical_features_indices = np.where(X.dtypes != np.float)[0]
cbmodel = CatBoostClassifier(learning_rate=1,
                             depth=6,
                             iterations=260,
                             loss_function='Logloss',
                             eval_set=(X_test, y_test),
#                             verbose=True,
                             eval_metric='Accuracy')
fit_model = cbmodel.fit(X_train, y_train, cat_features = categorical_features_indices)
predictions = cbmodel.predict(X_test)
print("Accuracy cb: %.2f%%" % (accuracy_score(y_test, predictions) * 100.0))


df = pd.read_csv('/Users/denispetruchik/Downloads/test.csv')
X_load = df.drop({'PassengerId'}, axis = 1)
X_pred = processX(X_load) 
predictions = random_forest.predict(X_pred)
StackingSubmission = pd.DataFrame({ 'PassengerId': df['PassengerId'],'Survived': predictions })
StackingSubmission.to_csv("Submission_rf.csv", index=False)

predictions = model.predict(X_pred)
StackingSubmission = pd.DataFrame({ 'PassengerId': df['PassengerId'],'Survived': predictions })
StackingSubmission.to_csv("Submission_xg.csv", index=False)
#
#predictions = cbmodel.predict(X_pred)
#submission = pd.DataFrame({ 'PassengerId': df['PassengerId'],'Survived': predictions.astype(int)})
#submission.to_csv("Submission_cb.csv", index=False)
#
#from sklearn import svm
#
#
#clf_svm = svm.SVC(class_weight='balanced')
#clf_svm.fit(X_train, y_train)
#predictions = clf_svm.predict(X_test)
#print("Accuracy svc: %.2f%%" % (accuracy_score(y_test, predictions) * 100.0))

