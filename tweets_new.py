ck = "HKHR55OCG8j3KyAlbSImU0TJG"
cks = "t5NTCOBtjxa0TSmYGZt1BsF6BQIroPR4NJsg29trvvH35ENNY0"
at = "1268190969767198722-jp1szCe5glvsR3ebqsF9SyEUYHIr5H"
ats = "3Llbw8gBUWJtHcyX9PPsVEnLqWS78SqZmbqGNksmq8qHG"

from tweepy import API 
from tweepy import Cursor
import numpy as np
import pandas as pd
import re
from tweepy import OAuthHandler #to authenticate based o credentials
from tweepy import Stream    #to establish a streaming session for downloading tweets in real time
from tweepy.streaming import StreamListener  #messages/tweets of the stream session are routed to it

class TwitterClient():
    """
    Class for streaming and processing live tweets.
    """
    def get_twitter_client_api(self):
        return self.twitter_client
    
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user
        
    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets    
    
    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets
    

# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(ck, cks)
        auth.set_access_token(at, ats)
        return auth

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
        stream = Stream(auth, listener)
        stream.filter(track=hash_tag_list)  #filtering the tweets based on the vairables passed in


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        return df

 
if __name__ == '__main__':

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
 
    api = twitter_client.get_twitter_client_api()
    screen_name="elonmusk"

    tweets = api.user_timeline(screen_name, count=20)

    #print(dir(tweets[0]))
    #print(tweets[0].retweet_count)
    cleaned = []
    df = tweet_analyzer.tweets_to_data_frame(tweets)
    for tweet in df['Tweets']:
        cleaned.append((tweet_analyzer.clean_tweet(tweet)))
        
    #extracting the data of the friends of the user we intended for
    friends = []
    for friend in Cursor(api.friends, screen_name).items(5):
         friends.append((friend.screen_name))
         
    #extracting the data of each of the friend of the user
    data = []
    for friend in friends:
        cleaned = []
        tt = api.user_timeline(friend, count=20)
        df = tweet_analyzer.tweets_to_data_frame(tt)
        for tweet in df['Tweets']:
            cleaned.append((tweet_analyzer.clean_tweet(tweet)))
        data.append(cleaned)
    

'''THE ML MODEL FOR THE PERSONALITY PREDICTION'''
from bs4 import BeautifulSoup
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

#reading the dataset
train = pd.read_csv('mbti_1.csv')
#what the target variables are
mbti = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
     
#counting what are the different types of personality available in the dataset
cnt_srs = train['type'].value_counts()
plt.figure(figsize=(12,4))
#plotting the count as bar plot
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Types', fontsize=12)
plt.show()

#function to clean the text available in the dataset
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text   #see
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

#cleaning the dataset values
train['clean_posts'] = train['posts'].apply(cleanText)

#applying naive bayes algorithm
scoring = {'acc': 'accuracy',
           'neg_log_loss': 'neg_log_loss',
           'f1_micro': 'f1_micro'}
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                         stop_words='english',
                         lowercase = True, 
                         max_features = 5000)

model_nb = Pipeline([('tfidf1', tfidf2), ('nb', MultinomialNB())])

results_nb = cross_validate(model_nb, train['clean_posts'], train['type'], cv=kfolds, scoring=scoring, n_jobs=-1)
print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_acc']),
                                                          np.std(results_nb['test_acc'])))

print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_f1_micro']),
                                                          np.std(results_nb['test_f1_micro'])))

print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_nb['test_neg_log_loss']),
                                                          np.std(-1*results_nb['test_neg_log_loss'])))

#predicting from the naive bayes model
model_nb.fit(train['clean_posts'], train['type'])
pred_all = model_nb.predict(train['clean_posts'])

#plotting the analysis of the tweets of the user entered
pred_user = model_nb.predict(df['Tweets'])
preds = pd.DataFrame({"col1": pred_user})
item_counts = preds["col1"].value_counts()
ss = screen_name + " personality Analysisis of 20 tweets"  
fig = plt.figure(figsize =(10, 7))
plt.pie(item_counts, labels = {'INFP','INFJ','INTP','INTJ','ISTP'})
plt.title(ss)
# show plot
plt.show()



















