from __future__ import division
import os
import time
import string
import pickle
import sys
import pandas
import numpy as np
import requests
import re
import csv
import mechanize
import cookielib
import smtplib
import ConfigParser

from bs4 import BeautifulSoup

from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split as tts
from sklearn.model_selection import KFold

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

BASEDIR = os.path.expanduser("~") + "/git/learnedleague-defender/"

def pull_matchday_questions(season, day):
    br = mechanize.Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

    br.addheaders = [('User-agent', 'Chrome')]
    br.open('https://learnedleague.com/ucp.php?mode=login')


    br.select_form(nr=0)
    br['username'] = LL_USERNAME
    br['password'] = LL_PASSWORD
    br.submit() 
    
    question_array = []
    
    r = br.open('https://learnedleague.com/match.php?' + str(season) + '&' + str(day)).read()
    soup = BeautifulSoup(r)
    qdivs = soup.find_all("div", class_="ind-Q20")
    for d in qdivs:
        question_text = ' '.join(d.text.split())
        s = "LL" + str(season)
        matchday = "MD" + "%02d" % (day,)
        question_number = question_text[0:2]
        text = question_text.split("- ", 1)[1]
        category = question_text.split(" -", 1)[0][3:]
        question_array.append([s, matchday, question_number, text, category])    


    return question_array

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

def pull_questions(csvfile):
    df = pandas.read_csv(csvfile, encoding = 'iso-8859-1')
    # df = df.loc[df['season'] != 'LL72'] 
    X = df['text'].tolist()
    y = df['category'].astype(str).tolist()
    return (X, y)

def build_model(X, y, classifier, verbose=True):

    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """

        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])

        model.fit(X, y)
        return model    
    
    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y) 

    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model, secs = build(classifier, X_train, y_train)

    if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))    

    if verbose: print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels.inverse_transform(model.classes_)

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    return model  

def show_most_informative_features(model, text=None, n=20):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)    

def download_question_history(username):

    outfile = BASEDIR + "user_data/" + username + ".csv"

    br = mechanize.Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

    br.addheaders = [('User-agent', 'Chrome')]
    br.open('https://learnedleague.com/ucp.php?mode=login')


    br.select_form(nr=0)
    br['username'] = LL_USERNAME
    br['password'] = LL_PASSWORD
    br.submit()    

    url = 'https://learnedleague.com/profiles/qhist.php?' + username

    print "Downloading question history from " + url

    r = br.open('https://learnedleague.com/profiles/qhist.php?' + username).read()
    soup = BeautifulSoup(r)

    tables = soup.find_all("table", class_="qh")

    seasons = []
    matchdays = []
    question_numbers = []
    corrects = []

    i = 0

    for table in tables:
        for c in table.find_all("tr"):
            cells = c.find_all("td", class_=re.compile("r|g"))
            if len(cells) > 0:
                matchday_details = cells[0].text
                tokens = matchday_details.split(" ")
                seasons.append(tokens[0])
                matchdays.append(tokens[1])
                question_numbers.append(tokens[2])
                gif = cells[2].find("img")["src"]
                corrects.append(gif != "/images/misc/reddot.gif")

    with open(outfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["season","matchday","question_number","correct"])
        for i in range(0, len(seasons)):
            writer.writerow([seasons[i], matchdays[i], question_numbers[i], corrects[i]])       

    return outfile

def pull_user_questions(csvfile):
    qd = pandas.read_csv(BASEDIR + "question_details.csv", encoding = 'iso-8859-1')
    df = pandas.read_csv(csvfile, encoding = 'iso-8859-1')
    df = df.merge(qd, on = ["season", "matchday", "question_number"])
    # v = df.loc[df['season'] == 'LL72']
    # df = df.loc[df['season'] != 'LL72']    
    X = df['text'].values
    y = df['correct'].values
    return (X, y)

@timeit
def build_and_evaluate_user_model(X, y):

    with open("categories.pickle", 'rb') as f:
        category_model = pickle.load(f)

    kf = KFold(n_splits=5)

    aucs = []

    for train, test in kf.split(X):

        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        features = category_model.predict_proba(X_train)

        user_model = LogisticRegression()
        user_model.fit(features, y_train)   
        
        y_pred = user_model.predict_proba(category_model.predict_proba(X_test))[:,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        aucs.append(metrics.auc(fpr,tpr))

    user_model.auc = sum(aucs)/len(aucs)

    features = category_model.predict_proba(X)
    user_model.fit(features, y)

    print(metrics.classification_report(y, user_model.predict(features))) 

    with open(BASEDIR + "user_models/" + username + ".pickle", 'wb') as f:
        pickle.dump(user_model, f)  

    print "Model written out to " + BASEDIR + "user_models/" + username + ".pickle" 

    return user_model    

def pull_today_questions(season, day):
    br = mechanize.Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

    br.addheaders = [('User-agent', 'Chrome')]
    br.open('https://learnedleague.com/ucp.php?mode=login')


    br.select_form(nr=0)
    br['username'] = LL_USERNAME
    br['password'] = LL_PASSWORD
    br.submit() 

    question_array = []

    r = br.open('https://learnedleague.com/match.php?' + str(season) + '&' + str(day)).read()

    soup = BeautifulSoup(r)

    for i in range(1,7):
        span = soup.find(id="q_field"+str(i))
        question_text = ' '.join(span.text.split())
        question_array.append(question_text)

    return question_array   

def score_questions(username, q1, q2, q3, q4, q5, q6):

    with open(BASEDIR + "categories.pickle", 'rb') as f:
        category_model = pickle.load(f)

    with open(BASEDIR + "user_models/" + username + ".pickle", 'rb') as f:
        user_model = pickle.load(f)
    
    features = category_model.predict_proba([q1, q2, q3, q4, q5, q6])

    return (user_model.predict_proba(features)[:,1], user_model.auc)

def send_email(questions, assigned_points, ranks):
    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(GMAIL_USERNAME,GMAIL_PASSWORD)
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "LL73 MD13"
    msg['From'] = GMAIL_USERNAME
    msg['To'] = GMAIL_USERNAME

    html = """\
    <html>
      <head></head>
      <body>
        <table>
            <tr><td>Question</td><td>Points</td><td>Rank</td></tr>"""
    
    for i in range(0, len(questions)):
        html += "<tr><td>" + questions[i] + "</td><td>" + str(assigned_points[i]) + "</td><td>" + str(ranks[i]) + "</td></tr>"
            
    html += """</table>
      </body>
    </html>
    """    
    
    html = html.encode("utf-8")

    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    
    server.sendmail(GMAIL_USERNAME, GMAIL_USERNAME, msg.as_string())
    server.quit()       

if __name__ == "__main__":

    # read username and password for the learned league site
    config = ConfigParser.ConfigParser()
    config.read("settings.ini")
    LL_USERNAME = config.get("LearnedLeague","username")
    LL_PASSWORD = config.get("LearnedLeague","password")   
    GMAIL_USERNAME = config.get("Gmail","username")
    GMAIL_PASSWORD = config.get("Gmail","password") 

    csvfile = BASEDIR + "question_details.csv"

    # this is hardcoded to fill in match days 16 - 24 from LL73!
    for matchday in range(9, 10):
        # Step 1: download the questions from match day and add to master question file
        questions = pull_matchday_questions(74, matchday)
        with open(csvfile, "a") as f:
            writer = csv.writer(f)
            # if add_newline:
            # writer.writerow("\n")
            for i in range(0, len(questions)):
                writer.writerow([questions[i][0].strip(), questions[i][1].strip(), questions[i][2].strip(), questions[i][3].encode("utf-8").strip(), questions[i][4].strip()])                  
    
    # Step 2: build category model
    X, y = pull_questions(csvfile)
    classifier = SGDClassifier(loss = "log")
    model = build_model(X, y, classifier)
    with open(BASEDIR + "categories.pickle", 'wb') as f:
        pickle.dump(model, f)   
    print("Model written out to {}".format("categories.pickle"))
    # print(show_most_informative_features(model))