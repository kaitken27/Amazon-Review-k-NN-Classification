#!/usr/bin/env python
# coding: utf-8

# # Amazon Review Classification with K-NN

# The purpose of this project was to create a machine learning model that would classify Amazon reviews by author. The [data set](https://archive.ics.uci.edu/ml/datasets/Amazon+Commerce+reviews+set) contained word frequency data from 1500 reviews (30 reviews from 50 authors). I also implemented a Term Frequency - Inverse Document Frequency (tf-idf) method on the data and reduced its dimensionality using PCA. $K$-Nearest Neighbor was the chosen method to classifiy the reviews.

# ## Data Preprocessing

# I will first clean the data by removing null values. Then I will apply the tf-idf method and reduce the dimensionality of of the data using Principal Component Analysis.

# In[91]:


import numpy as np
import pandas as pd


# In[92]:


raw = pd.read_csv('Amazon_initial_50_30_10000.csv') #imports the data
raw


# We can see that the data consists of common strings found in the reviews on the columns and the review numbers on the rows. Each entry contains the amount of times that particular string appears in the the review with null values occupying columns on the right.

# In[93]:


raw.dropna(axis=1, how='all', inplace=True) #drops all columns with strictly null values
raw


# Now all the columns with strictly null values are gone and the authors of the reviews are stored in the rightmost column.

# In[94]:


from sklearn import preprocessing as sklpp

authors = raw.iloc[:, -1] #creates a dataframe of the authors
label_encoder = sklpp.LabelEncoder()
authors = label_encoder.fit_transform(authors) #label encodes the authors with integers from 0 to 49.

freq = raw.iloc[:, :-1] #creates a dataframe with all the string frequencies


# Now I'm going to remove columns containing "stop words" (commonly used words that will not help classify the review such as "a", "the", etc.)

# In[95]:


from nltk.corpus import stopwords

stop_words = stopwords.words('english') #imports a set of stop words from nltk


# In[96]:


#removes all stop word columns from dataset
for i in freq.columns:
    if i.lower() in stop_words:
        freq = freq.drop(i, axis=1)
freq


# We also have some columns in the freq dataframe that have a combination of apostrophes and backslashes that I would like to remove. These strings are probably a result of some technique used by the creators of the data set when compiling the data. These strings are most likely not indicative of the author.

# In[97]:


freq = freq[freq.columns.drop(list(freq.filter(regex=r"\\")))] #removes all columns with a backslash in name
freq


# I am now going to implement the Term Frequency - Inverse Document Frequency method. This method is a very effective and commonly used term-weighting scheme. It refelects the importance of a particular word in classifying a document, while also accounting for the fact that some words appear frequently in general and therefore have less importance than words that appear in fewer documents in the corpus.

# In[98]:


#creating the term frequency dataframe - divides each value by the sum of the entries in its row
tf = (freq.T / freq.T.sum()).T

#creating the inverse document frequency dataframe
num_reviews = len(freq.index) #total number of reviews is 1500 in this case
num_appearances = freq.gt(0).sum() #amount of documents in which each string appears
idf = np.log(num_reviews/num_appearances)

#creating the finalized tf-idf dataframe, ready for dimensionality reduction
tf_idf = tf*idf
tf_idf


# I am now going to do Principal Component Analysis on the tf-idf dataframe to reduce the amount of features that we are dealing with.

# In[99]:


from sklearn.decomposition import PCA

pca = PCA(n_components=.90) #this retains 90% of the information in the original tf-idf dataframe
pca.fit(tf_idf) #calculates the mean and variance of each column in the tf-idf dataframe
tf_idf = pca.transform(tf_idf) #manipulates data to have zero mean and unit variance
tf_idf = pd.DataFrame(tf_idf) #this is the finalized data, ready for K-NN
tf_idf


# We are now dealing with much fewer features than 9755. This data is now sufficiently preprocessed.

# ## K-NN Classification

# I will first split the data into into training and test data, with 30% of the data being used to test the learning method. Then I will determine the best choice of $K$ and implement the $K$-$NN$ method and visualize the data.

# In[100]:


import time
import matplotlib.pyplot as plt
import seaborn as sns


# In[101]:


from sklearn.model_selection import train_test_split

#splits the data into test and training data with 30% used for test.
X_train, X_test, y_train, y_test = train_test_split(tf_idf,authors,test_size=0.30)


# I am now going to determine the best $K$ value for $K$-$NN$.

# In[102]:


from sklearn.neighbors import KNeighborsClassifier

k_pick_start = time.process_time() #start time for picking K

error_rate = [] #creates empty array for future storage of error rates for K-NN with different K values.

#calcuates the rate of misclassification for each K and stores it in the error_rate array
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test) #runs K-NN for K values ranging from 1 to 20.
    error_rate.append(np.mean(pred_i != y_test))

fig = plt.figure(figsize=(7.5,5), dpi=100)

axes = fig.add_axes([.2,.2,1,1])

#plots the error_rate array
axes.plot(range(1,20), error_rate, color='black', marker='s',
          markerfacecolor='red', markersize=8)
axes.set_title('Error Rate vs. K', fontsize=20, fontweight='bold')
axes.set_xlabel('K', fontsize=15)
axes.set_ylabel('Error Rate', fontsize=15)
axes.xaxis.set_ticks(np.arange(0, 21, 1))

plt.show(fig)

k_pick_end = time.process_time() #end time for picking K


# This shows that the best choice of $K$ is 1 because it minimizes the rate of misclassification.

# In[103]:


knn_start = time.process_time() #start time for K-NN method

knn = KNeighborsClassifier(n_neighbors=1) #choose K=1
knn.fit(X_train,y_train) #runs the method on X_train and y_train
pred = knn.predict(X_test) #stores the predictions of the K-NN method

knn_end = time.process_time() #end time for K-NN method


# In[104]:


from sklearn.metrics import confusion_matrix, accuracy_score

fig2 = plt.figure(figsize=(7.5,5), dpi=100)

cm_heatmap = sns.heatmap(confusion_matrix(y_test,pred), xticklabels=10, yticklabels=10, cmap="Reds")

cm_heatmap.set_title('Confusion Matrix Heatmap for K=1', fontsize=15, fontweight='bold')
cm_heatmap.set_xlabel('Predicted Author', fontsize=15)
cm_heatmap.set_ylabel('Actual Author', fontsize=15)
cm_heatmap.set_ylim(50,0)
cm_heatmap.xaxis.set_ticks_position('top')

plt.show(fig2)


# Here, I opted to display the confusion matrix as a heatmap because the original matrix is 50 by 50 and hard to interpret. This heatmap shows that the learning method was effective because the darker squares are almost exclusively on the diagonal which represents correct classifications.
#
# Furthermore, the squares not on the diagonal are primarily light reds, which shows that the model was not prone to specific misclassifications (i.e. frequently misclassifying author $x$ as author $y$ specifically).

# In[105]:


print('K-NN classification Accuracy with K=1: '
      + str(round(accuracy_score(y_test, pred)*100,2))+ '%')


# In[106]:


k_pick_runtime = k_pick_end - k_pick_start
knn_runtime = knn_end - knn_start

runtime = pd.DataFrame({'Runtime': [k_pick_runtime, knn_runtime]},
                        index=['Picking K', 'Running K-NN'])
runtime


# ## Conclusion

# Overall, $K$-$NN$ was a good method to use on this dataset. The program was over 30 percent accurate, which is over 15 times more effective than standalone guessing which, would only be 1/50=2 percent effective. The method of choosing $K$ was not exceptionally fast, but it was not unbareably slow either. If there were significantly more features in the dataset, $K$-$NN$ may not have been an appropriate technique.
#
# Removing the columns with the backslashes in the name improved the accuracy by about 5%, which highlights the importance of scrutinizing your dataset before mindlessly implementing preprocessing or ML techniques on it.
#
# Another thing worth noting is that doing PCA on the dataset was very neccessary. Without PCA, picking $K$ took an unreasonanly long time and the classification accuracy was actually worse.
#
# The effectiveness of this method outways its shortcomings in computing time. It would be interesting to compare the accuracies and runtimes of other classification techniques on this dataset.
