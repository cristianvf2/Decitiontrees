#!/usr/bin/env python
# coding: utf-8

# # Fundamentals of Machine Learning

# In[1]:


print("hello world")


# ## Machine learning classification tasks

# Importing modules

# In[2]:


import numpy as np #linear algebra
import pandas as pd #dataframes
import sklearn.preprocessing #raw data preprocessing
import sklearn.model_selection #grid search + cross validation
import sklearn.ensemble #random forest
import sklearn.tree #decision trees
import sklearn.linear_model #logistic regression + perceptron
import sklearn.svm #support vector machines
import sklearn.neighbors #k-nearest neighbors
import sklearn.neural_network #multilayer perceptron
import matplotlib.pyplot as plt #visualization
import xgboost as xgb #extreme gradient boosting


# Reading the data from a csv file and saving it into a dataframe

# In[3]:


data = pd.read_csv('exampleObesityClassReg.csv')


# ### Data Preprocessing

# Data exploration

# In[9]:


data.sample(10)


# Checking if the dataset is balanced

# In[15]:


data.tertile.value_counts()


# In[53]:


##########

# what if the dataset is unbalanced?, describe the problems that can generate a training session whith unbalanced data
# and code the solution to balance the data

# HINT:
# count_class_0, count_class_1 = dataframe['label'].value_counts() # class 0 is the predominant
# df_class_0 = dataframe[dataframe['label'] == 0]
# df_class_1 = dataframe[dataframe['label'] == 1]
# df_class_0_under = df_class_0.sample(count_class_1)
# dataframe_balanced = pd.concat([df_class_0_under, df_class_1], axis=0)

##########


# Describing the shape of the dataframe

# In[54]:


data.shape


# Creating a variable which only contains the predictor features

# In[55]:


features = data[data.columns.difference(['prevalence','tertile','country_name'])]


# Saving the feature names into a variable

# In[56]:


feature_names = features.columns


# If one would like to consider continents the way to do it is by encoding the categorical features into dummy variables like (in R one can use "as.factor"):
# * continent_Asia
# * continent_Europe
# * ...

# In[57]:


pd.get_dummies(features)


# But does it have sense to use the continents as predictor? Let's use only food as predictors. 

# In[58]:


features = features[features.columns.difference(['continent'])]


# In[59]:


features.sample(3)


# Creating a variable of the categorical labels of the dataset

# In[60]:


labels_categorical = data.tertile


# Normalizing the numerical variables into a min max scaler where the maximum value is transformed into a 1 and the minimum value is 0

# In[61]:


minMaxScaler = sklearn.preprocessing.MinMaxScaler()
features = minMaxScaler.fit_transform(features)


# In[62]:


features[:1,:]


# Splitting the data into training and testing subsets

# In[63]:


features_train, features_test, labels_categorical_train, labels_categorical_test = sklearn.model_selection.train_test_split(
    features,
    labels_categorical,
    test_size=0.30,
    random_state = 55,
    stratify = labels_categorical #preserving the probability distribution of the original dataset
)


# In[64]:


features_train


# ### Machine Learning Data Analysis

# #### Single Decision tree

# Training a single decision tree with a max depth of 2

# In[65]:


decisionTree = sklearn.tree.DecisionTreeClassifier(max_features=None, max_depth=2)
decisionTree.fit(features_train,labels_categorical_train)


# Plotting the decision tree

# In[66]:


ax = sklearn.tree.plot_tree(decisionTree,
                       feature_names = feature_names
                      )
plt.show()


# Training a single decision tree with a without specifying the maximum depth

# In[67]:


decisionTree_full = sklearn.tree.DecisionTreeClassifier(max_features=None)
decisionTree_full.fit(features_train,labels_categorical_train)


# Plotting the tree

# In[68]:


ax = sklearn.tree.plot_tree(decisionTree_full,
                       feature_names = feature_names
                      )
plt.savefig('tree.pdf') #saving the tree plot as pdf
plt.show()


# ### Ensemble of Trees

# Training a random forest classifier with 100 trees

# In[69]:


classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=15)


# In[70]:


classifier.fit(features_train,labels_categorical_train)


# Predicting the labels from the test features subset

# In[71]:


predictions = classifier.predict(features_test)


# Printing the confusion matrix between the true and predicted values

# In[72]:


print(sklearn.metrics.confusion_matrix(predictions, labels_categorical_test))


# Extracting some performance metrics from the prediction

# In[73]:


print(sklearn.metrics.classification_report(predictions, labels_categorical_test))


# Returning the probability of being classified as each class along with their true label

# In[96]:


predictions_probabilities = classifier.predict_proba(features_test)
predictions_probabilities = pd.DataFrame(predictions_probabilities)
predictions_probabilities_labels = pd.concat([predictions_probabilities,pd.DataFrame(list(labels_categorical_test))], axis=1)
predictions_probabilities_labels.columns = ['class_1_probability','class_2_probability','class_3_probability','true_class']
predictions_probabilities_labels


# Extracting the variable importance list from the random forest classifier

# In[26]:


vil = pd.DataFrame(list(zip(feature_names,classifier.feature_importances_)),
                   columns=['feature','mean_gini_decrease']
                  ).sort_values(by='mean_gini_decrease', ascending=False)


# In[27]:


vil.head()


# Plotting the variable importance list sorted by the mean decrease gini

# In[28]:


ax = plt.barh(data = vil.sort_values(by='mean_gini_decrease').tail(15), 
         y='feature', 
         width = 'mean_gini_decrease'
        )
plt.show()


# Averaging the VIL from 50 random forest runs

# In[29]:


rf_vil  = []
for i in range(50):
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    rf = rf.fit(features_train,labels_categorical_train)
    rf_vil.append(rf.feature_importances_)
mean_vil = np.mean(rf_vil, axis = 0)


# In[30]:


mean_vil = pd.DataFrame(list(zip(feature_names,mean_vil)),
                   columns=['feature','mean_gini_decrease']
                  ).sort_values(by='mean_gini_decrease', ascending=False)


# In[31]:


mean_vil.head()


# In[32]:


ax = plt.barh(data = mean_vil.sort_values(by='mean_gini_decrease').tail(15), 
         y='feature', 
         width = 'mean_gini_decrease'
        )
plt.show()


# ### Comparison Between Diferent Machine Learning Models and Hyperparameters

# Selecting some machine learning training algorithms, training each one and using cross validation to retrieve their performance

# In[42]:


models = []

models.append(("LogisticRegression",sklearn.linear_model.LogisticRegression()))
models.append(("SVC",sklearn.svm.SVC()))
models.append(("RandomForest",sklearn.ensemble.RandomForestClassifier()))
models.append(("KNeighbors",sklearn.neighbors.KNeighborsClassifier()))
models.append(("MLPClassifier",sklearn.neural_network.MLPClassifier()))
models.append(("XGBoost",xgb.XGBClassifier()))
models.append(("DecisionTree",sklearn.tree.DecisionTreeClassifier()))


results = []
names = []
for name,model in models:
    result = sklearn.model_selection.cross_val_score(model, 
                             features, 
                             labels_categorical,  
                             cv=10, 
                             scoring='accuracy',
                             n_jobs=-1
                            )
    names.append(name)
    results.append(result)


# List of compared models along with their hyperparameters

# In[43]:


models


# Box plot of the results of each model

# In[44]:


plt.boxplot(results,labels = names)
plt.xticks(rotation=45)
plt.show()


# In[45]:


##########

#compare the models based on their f1-score, what is the problem?

##########


# Grid search for selection of the best hyperparameters

# In[46]:


# HIGH PROCESSOR INTENSIVE TASK (~5 minutes calculations of a 12 cores machine)

# parameters = {'n_estimators': list(range(260,360,20)),
#               'max_features': ['auto', 'sqrt'],
#               'max_depth': list(range(20,100,20))+[None],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4],
#               'bootstrap': [True, False]
# }
# grid_Search = sklearn.model_selection.GridSearchCV(sklearn.ensemble.RandomForestClassifier(), 
#                                                    parameters, 
#                                                    n_jobs=-1,
#                                                    cv = 3
#                                                   )

# grid_Search.fit(features_train,labels_categorical_train)


# In[47]:


#low intensive grid search
parameters = {'n_estimators': [100,150,200]
}
grid_Search = sklearn.model_selection.GridSearchCV(sklearn.ensemble.RandomForestClassifier(), 
                                                   parameters, 
                                                   n_jobs=-1,
                                                   cv = 3
                                                  )

grid_Search.fit(features_train,labels_categorical_train)


# In[49]:


grid_Search.best_params_


# In[50]:


pd.DataFrame(grid_Search.cv_results_).head(10)


# One can apply predict to gridSearch and automatically calculates the given model with the best hyper-parameters. 

# In[57]:


predictions = grid_Search.predict(features_test)


# In[58]:


print(sklearn.metrics.confusion_matrix(predictions, labels_categorical_test))


# In[59]:


print(sklearn.metrics.classification_report(predictions, labels_categorical_test))


# In[60]:


##########

#make your own grid search for other training algorithm

##########

