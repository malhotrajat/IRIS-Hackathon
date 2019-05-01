
# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import VotingClassifier
from textblob import Word
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import voting_classifier
from sklearn.metrics import accuracy_score  
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import random
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler 

# Working directory
os.chdir('C:\\Users\\Parikshit_verma\\Documents\\hackathon')

# Fucntions
def pre_process(train):
    
    # Data Pre-processing
    train = train[train['ReviewText'].notnull()] # Correct data types and change to string
    train['word_count'] = train['ReviewText'].apply(lambda x: len(str(x).split(" "))) # Generate word count
    def avg_word(sentence): 
        words = sentence.split() 
        return (sum(len(word) for word in words)/len(words))
    train['avg_word'] = train['ReviewText'].apply(lambda x: avg_word(x)) # Average number of words
    
    stop = stopwords.words('english')
    train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop )) # Remove stop words
    train['SpecialChar'] = train['ReviewText'].apply(lambda x: len([x for x in x.split() if x.startswith('#')])) # Identify special characters
    train['numerics'] = train['ReviewText'].apply(lambda x: len([x for x in x.split() if x.isdigit()])) # Identify numerics
    
    train['ReviewText'] = train['ReviewText'].str.replace('[^\w\s]','') # Remove punctuations
    
    st = PorterStemmer()
    train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) # Stemming
    train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) # Lemmetization
    
    return train

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
 
def predictions(test1,testB,clf):
    test1 = pre_process(test1)
    testB = pre_process(testB)
    
    test1 = tfidf_test(test1['ReviewText'])
    testB = tfidf_test(testB['ReviewText'])
    
    test1 = clf.predict(test1)
    testB = clf.predict(testB)
    
    test1 = pd.DataFrame({'Rating':test1})
    testB = pd.DataFrame({'Rating':testB})
    
    test1.to_csv("test1.csv")
    testB.to_csv("testB.csv")
    
    return test1,testB    

# Read dataset
train = pd.read_csv('train.csv',nrows = 100000)
test1 = pd.read_csv('test1_generic_reviews.csv')
testB = pd.read_csv('testB_dell_reviews.csv')    

# Train dataset preprocessing and vectorization
train = pre_process(train)
Review, Rating = stratified_sampling(train,23)
Review = tfidf(Review)

# Model building
train_Review, test_Review, train_Rating, test_Rating = train_test_split(Review, Rating, test_size=0.30, random_state=42)

# Naive Bayes
nbclf = MultinomialNB().fit(train_Review, train_Rating)

# logistics Regresion
lclf = LogisticRegression().fit(train_Review, train_Rating)

# Random forest
rf = RandomForestClassifier()
params_rf = {'n_estimators': [50, 100, 200]}
rf_gs = GridSearchCV(rf, params_rf, cv=5)
rf_fit = rf_gs.fit(train_Review,train_Rating)

# XgBoost
xg = xgb.XGBClassifier()
xg_fit = xg.fit(train_Review,train_Rating) 

#create a dictionary of our models
estimators=[('log', lclf), ('rf', rf_fit), ('nb',nbclf)]
ensemble = VotingClassifier(estimators, voting='hard') 
ensemble = ensemble.fit(train_Review,train_Rating)

# Test predictions
test1,testB = predictions(test1,testB,ensemble)




