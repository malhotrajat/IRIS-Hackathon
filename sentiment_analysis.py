# -----------------------------------------------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
import random
import imblearn
import pickle
import os
from scipy import sparse
import warnings
import datetime as dt

from textblob import Word
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler 

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import voting_classifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------------------------------------------

os.chdir('/home/cdsw/Analysis_PV')

# -----------------------------------------------------------------------------------------------------------------
# Fucntions
# -----------------------------------------------------------------------------------------------------------------

def pre_process(train):
    
    # Data Pre-processing
    stop = stopwords.words('english')
    train = train[train['ReviewText'].notnull()] # Correct data types and change to string   
    train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop )) # Remove stop words    
    train['ReviewText'] = train['ReviewText'].str.replace('[^\w\s]','') # Remove punctuations  
    st = PorterStemmer()
    train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) # Stemming
    train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) # Lemmetization
    
    return train

def add_features(df):
    df['ReviewText'] = df['ReviewText'].apply(lambda x:str(x))
    df['total_length'] = df['ReviewText'].apply(len)
    df['capitals'] = df['ReviewText'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.ReviewText.str.count('\S+')
    df['num_unique_words'] = df['ReviewText'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words'] 
    return df
  
def stratified_sampling(data, rand_seed):
  ''' Function to over and under sample the dataset based on target variable, concatenate them and return
  Input dataframe needs to have 2 columns named ReviewText containing plain text review and Rating column
  '''
  random.seed(rand_seed)

  Review = data['ReviewText'].to_frame()
  Rating = data['Rating'].to_frame()

  rus = RandomUnderSampler(return_indices=True)
  Review_rus, Rating_rus, id_rus = rus.fit_sample(Review, Rating)

  ros = RandomOverSampler()
  Review_ros, Rating_ros = ros.fit_sample(Review, Rating)

  Review = pd.DataFrame(Review_rus).append(pd.DataFrame(Review_ros), ignore_index=True)
  Rating = pd.DataFrame(Rating_rus).append(pd.DataFrame(Rating_ros), ignore_index=True) 
  
  Review.columns = ['ReviewText']
  
  return Review, Rating

def tfidf(Review) :
    Review.columns = ['ReviewText']    
    # TF IDF vectorization
    vectorizer = TfidfVectorizer(max_df = 0.999,
                                 min_df = 0.001,
                                 lowercase=True, 
                                 ngram_range=(1,2),
                                 analyzer = "word")
    Review = vectorizer.fit_transform(Review['ReviewText'])
    pickle.dump(vectorizer.vocabulary_,open("vocab.pkl","wb"))
    
    return Review

def tfidf_test(test):
    transformer = TfidfTransformer()
    vocab = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("vocab.pkl", "rb")))
    test.columns = ['ReviewText']
    test = transformer.fit_transform(vocab.fit_transform(test))
    
    return test

def test(test,test_Rating,clf):
    test = clf.predict(test)
    score = accuracy_score(test_Rating,test)
    return score
    
def predictions(test1,testB,clf):
    
    test1 = pre_process(test1)
    test1 = tfidf_test(test1['ReviewText'])
    test1 = clf.predict(test1)
    test1 = pd.DataFrame({'Rating':test1})
    test1.to_csv("test1.csv",index = 0)
    
    testB = pre_process(testB)    
    testB = tfidf_test(testB['ReviewText'])    
    testB = clf.predict(testB)    
    testB = pd.DataFrame({'Rating':testB})    
    testB.to_csv("testB.csv", index = 0)
    
    return test1,testB    

# -----------------------------------------------------------------------------------------------------------------
# Read dataset
# -----------------------------------------------------------------------------------------------------------------
    
train = pd.read_csv('train.csv',nrows = 500000)
test1 = pd.read_csv('test1_generic_reviews.csv')
testB = pd.read_csv('testB_dell_reviews.csv')    

# -----------------------------------------------------------------------------------------------------------------
# Train & Test data initialization
# -----------------------------------------------------------------------------------------------------------------

train = pre_process(train)
Review, Rating = stratified_sampling(train,23)
print("Creating TF IDF vectors")
Review = tfidf(Review)

# Train-Test split
print("Creating Train and test splits")
train_Review, test_Review, train_Rating, test_Rating = train_test_split(Review, Rating, test_size=0.30, random_state=42)
sparse.save_npz("train_Review.npz", train_Review)
train_Rating.to_pickle("./train_Rating.pkl")

# -----------------------------------------------------------------------------------------------------------------
# Load Pre-processed data
# -----------------------------------------------------------------------------------------------------------------

train_Rating = pd.read_pickle("./train_Rating.pkl")
train_Review = sparse.load_npz("train_Review.npz")

# -----------------------------------------------------------------------------------------------------------------
# Parameter grid search
# -----------------------------------------------------------------------------------------------------------------

# logistics Regression
"""
parameter_candidates = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}]
lclf = GridSearchCV(estimator=LogisticRegression(penalty = 'l2', solver='lbfgs', multi_class='multinomial'), 
                    param_grid=parameter_candidates, n_jobs=-1)
lclf = lclf.fit(train_Review, train_Rating)
print('Best score:', lclf.best_score_) 
print('Best alpha:',lclf.best_estimator_.C)  
"""

# -----------------------------------------------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------------------------------------------

# Naive Bayes
print("Naive Bayes")
nbclf = MultinomialNB(alpha = 0.2).fit(train_Review, train_Rating)

# logistics Regresion
print("Logistics Regression")
lclf = LogisticRegression(penalty = 'l2', C = 0.8).fit(train_Review, train_Rating)

# Random forest
print("Random Forest")
rfclf = RandomForestClassifier()
rf_fit = rfclf.fit(train_Review, train_Rating)

#create a dictionary of our models
print("Ensemble voting")
estimators=[('log', lclf), ('rf', rf_fit), ('nb',nbclf)]
ensemble = VotingClassifier(estimators, voting='hard') 
ensm = ensemble.fit(train_Review,train_Rating)

# -----------------------------------------------------------------------------------------------------------------
# Test predictions
# -----------------------------------------------------------------------------------------------------------------

# Test set predictions and accuracy
test_score = test(test_Review,test_Rating,ensm)
print(test_score)

# Unseen data predictions
test1_Rating,testB_Rating = predictions(test1,testB,ensm)

# -----------------------------------------------------------------------------------------------------------------
# End
# -----------------------------------------------------------------------------------------------------------------
print(dt.datetime.now())
