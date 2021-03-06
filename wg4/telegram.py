# basic telegram bot
# https://www.codementor.io/garethdwyer/building-a-telegram-bot-using-python-part-1-goi5fncay
# https://github.com/sixhobbits/python-telegram-tutorial/blob/master/part1/echobot.py

import json
import random
import requests
import time
import urllib
import glob
import os
import numpy as np
import re

#from cobe.brain import Brain
from markov_norder import Markov
from config import token
from gensim import models, corpora
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize

# python3: urllib.parse.quote_plus
# python2: urllib.pathname2url

GREETINGS = ["Hello", "Hi", "Hey", "Greetings", "Hi there", "Hey there"]
ENDCONVERSATION = ["Thanks","Thank you","Okay"]

URL_RE = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.IGNORECASE|re.DOTALL)
TOKEN = token # don't put this in your repo! (put in config, then import config)
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))#"""D:/Documents/GitHub/CCMWLI-2017/wg4"""
DATA_DIR = os.path.join(ROOT_DIR, 'data_hede')
TRAIN_DIR = os.path.join(DATA_DIR, 'topics')

class Telegram:

    def __init__(self):
        # self.order = 1
        self.name = requests.get(URL + 'getme').json()['result']['username']
        self.stopwords = set(stopwords.words('english')).union(set(['one', 'say', 'also', 'could', 'would', 'should', 'shall']))
        self.bots = []
        self.topics = glob.glob(os.path.join(TRAIN_DIR, '*'))

        num_topics = len(self.topics)
        model_file = 'lda_{}.bin'.format(num_topics)
        model_path = os.path.join(DATA_DIR, model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                'Model not found: {}. Be sure to train the LDA model before running the bot.'.format(model_file))
        else:
            self.model = models.ldamodel.LdaModel.load(model_path)

        ####### BASE MARKOV NET ############
        # create one network per topic
        for i, topic in enumerate(self.topics):
            self.bots.append(Markov(topic))
            self.bots[i].generate_table(topic)

        ######## COBE's BRAIN IMPLEMENTATION ###########
        # create one network per topic
        # self.bots = [Markov(self.order) for topic in range(num_topics)]
        # self.bots = []
        # for i, topic in enumerate(self.topics):
        #     print('Generating bot for topic %s...' % i)
        #
        #     bot_file = 'bot_%s.brain' %i
        #     bot = Brain(bot_file, order=self.order)
        #     if os.path.isfile(bot_file):
        #         print('Brain %s already there. Skipping it.' %bot_file)
        #     else:
        #         bot.learnfile([topic])
        #
        #     self.bots.append(bot)

                # for i, topic in enumerate(topics):
        #     self.bots[i].generate_table(topic)

    def get_url(self, url):
        response = requests.get(url)
        content = response.content.decode("utf8")
        return content

    def get_json_from_url(self, url):
        content = self.get_url(url)
        js = json.loads(content)
        return js

    def get_updates(self, offset=None):
        url = URL + "getUpdates"
        if offset:
            url += "?offset={}".format(offset)
        js = self.get_json_from_url(url)
        return js

    def get_last_update_id(self, updates):
        update_ids = []
        for update in updates["result"]:
            update_ids.append(int(update["update_id"]))
        return max(update_ids)

    def echo_all(self, updates):
        for update in updates["result"]:
            text = update["message"]["text"]
            chat = update["message"]["chat"]["id"]
            self.send_message(text, chat)

    def handle_updates(self, updates):
        for update in updates["result"]:
            try:
                text = update["message"]["text"]
                chat = update["message"]["chat"]["id"]

                self.send_message('Let me think...', chat)

                mess = self.get_reply(text)

                # mess = bot.generate_output(max_words=100, newline_after=None, seed=text)
                self.send_message(mess, chat)

            except KeyError:
                pass

    def get_reply(self, text):

        response = ''
        text = text.lower()

        if any(greet.lower() in text.split() for greet in GREETINGS):
            response = random.choice(GREETINGS) + '!\n'
            if len(text.split()) == 1:
                return response

        if "bye" in text:
            response = "Bye, I hope to see you again!!"
            if len(text.split()) == 1:
                return response

        if any(endcon.lower() in text.split() for endcon in ENDCONVERSATION):
            response += "I hope I could help you."

        if "your name" in text:
            response += "My name is {}".format(self.name)

        else:
            query_topic = self.get_query_topic(text)
            # print('Topic', query_topic)
            bot = self.bots[query_topic]
            response += bot.generate_output(max_words=100, newline_after=None, seed=word_tokenize(text))
            # return bot.reply(text, loop_ms=5000, max_len=100)

        return response

    def get_last_chat_id_and_text(self, updates):
        num_updates = len(updates["result"])
        last_update = num_updates - 1
        text = updates["result"][last_update]["message"]["text"]
        chat_id = updates["result"][last_update]["message"]["chat"]["id"]
        return (text, chat_id)

    def send_message(self, text, chat_id):
        text = urllib.parse.quote_plus(text) # (python3) #urllib.pathname2url(text) #
        url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
        self.get_url(url)

    def run(self):
        last_update_id = None
        print('Running...')
        while True:
            updates = self.get_updates(last_update_id)
            if len(updates["result"]) > 0:
                last_update_id = self.get_last_update_id(updates) + 1
                print('Udpate received: {}'.format(updates))
                self.handle_updates(updates)
            time.sleep(0.5)

    # Just get the most likely topic for the query according to the model
    def get_query_topic(self, query):
        query_ = self.model.id2word.doc2bow(self.preprocess(query))
        query_topic = np.argmax([score for t, score in self.model[query_]])
        return query_topic

    def get_query_topics(self, query):
        query_ = self.model.id2word.doc2bow(self.preprocess(query))
        return self.model.get_document_topics(query_)

    def get_topic_seed(self, topic, bot):
        seed,_,_ = np.where()[0]
        return bot.table[seed]

    def preprocess(self, text, stemming=True):
        text = text.lower()
        text = URL_RE.sub('', text)  # remove http links
        text = re.sub(r'([a-z0-9])\/([a-z0-9])', r'\1 \2', text)  # convert 'a/b' to 'a b'
        text = re.sub('[^a-z\s]', '', text)  # get rid of noise (punctuation, symbols, etc.)

        # for some reasons, single letters would end up in a topic (perhaps something to do with logical formulae).
        # Just remove everything below three-chars length
        if stemming:
            stemmer = SnowballStemmer(language='english')
            text = [stemmer.stem(w) for w in word_tokenize(text) if w not in self.stopwords and len(w) > 3]
        else:
            text = word_tokenize(text)

        return text


if __name__ == '__main__':
    Telegram().run()
    # telegram = Telegram(order=5)
    # question = 'What is the meaning of everything?'
    #
    # print('Let me think...')
    # print(telegram.get_reply(question))
