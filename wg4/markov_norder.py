import sys, random, collections, os, re
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

# from gensim.models import Word2Vec
# from gensim.models.keyedvectors import KeyedVectors

# ROOT_DIR = os.path.abspath(os.path.dirname(__file__))#"""D:/Documents/GitHub/CCMWLI-2017/wg4"""
# DATA_DIR = os.path.join(ROOT_DIR, 'data_hede')
# W2V_FILE = os.path.join(DATA_DIR, 'glove.6B.300d.txt.word2vec')

# Since we split on whitespace, this can never be a word
NONWORD = "\n"


class Markov():
    """
    This basically parses the text through a fixed-size sliding window (the order param).
    A dict stores a mapping between n-grams of size=order and the word coming right next.
    At generation time, the dictionary is queried for possible words in the following loop:
        - first, the most probable next word is chosen given the last seen n-gram
        - second, this word is added to the list of seen word to generate the next n-gram.
          This is equivalent to a one-step slide of the window.
        - the loop is repeated
    """
    def __init__(self, topic):
        self.order = 1 # Default order set to 1, but the order is updated based on the seed
        self.topic = topic
        self.table = collections.defaultdict(list)
        self.seen = collections.deque([NONWORD] * self.order, self.order)
        self.vectorizer = None
        self.vocabulary = []
        # self.word_vectors = KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)

    def generate_tfidf(self, filename):
        topic_corpus = open(filename, 'r', encoding='utf-8').readlines()
        topic_corpus_tokens = [word_tokenize(line) for line in topic_corpus if line is not '\n']
        self.vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,1))
        self.vectorizer.fit(topic_corpus_tokens)
        self.vocabulary = self.vectorizer.get_feature_names()

    def get_tfidf(self, word_list):
        query_matrix = self.vectorizer.transform([word_list], copy=True)
        tfidf_scores = {}
        for word in word_list:
            if word in self.vocabulary:
                tfidf_scores[word] = query_matrix[0, self.vocabulary.index(word)]
            else:
                tfidf_scores[word] = 0
        return tfidf_scores

    # Generate table
    def generate_table(self, filename):
        for line in open(filename, 'r', encoding='utf-8'):
            line = line.lower()
            for word in word_tokenize(line):
                if word != NONWORD:
                    self.table[tuple(self.seen)].append(word)
                    self.seen.append(word)
        self.table[tuple(self.seen)].append(NONWORD)  # Mark the end of the file

    # table, seen = generate_table("gk_papers.txt")

    def update_order(self, order):
        if self.order is not order:
            self.order = order
            self.seen = collections.deque([NONWORD] * self.order, self.order)
            self.generate_table(self.topic)

    # Generate output
    def generate_output(self, max_words=100, newline_after=10, seed=None):
        output = ''

        while True:
            if seed:
                print("Seed: ", seed)
                # Update the order and the table based on the lenght of the seed
                self.update_order(len(seed))

                print("New order", self.order)

                self.seen = collections.deque(seed, self.order)
                self.seen.extend(seed)
                # if seed not doesn't exist:
                try:
                    word = random.choice(self.table[tuple(self.seen)])
                except IndexError as e:
                    print("Seed is not found in the corpus.")

                    # If the vectorizer is not created so far, create it
                    if not self.vectorizer:
                        self.generate_tfidf(self.topic)

                    # Get the TF-IDF scores of the words in the list
                    tfidf_scores = self.get_tfidf(seed)

                    # Find out if the first or the last word in the seed has lower TF-IDF score
                    min_value = min([tfidf_scores[seed[0]], tfidf_scores[seed[-1]]])
                    word_min_tfdf = [key for key, value in tfidf_scores.items() if value == min_value][0]

                    # Remove the one with the lower TF-IDF score
                    seed.remove(word_min_tfdf)
                    print('New version seed:', seed)
                else:
                    # If the seed found in corpus,
                    # But the order is less then 7, add a random word to the seed from table
                    if self.order < 7:
                        word = random.choice(self.table[tuple(self.seen)])
                        seed.append(word)
                    else:
                        # If the order is also 7 or more, prepare self.seen
                        self.update_order(len(seed))
                        self.seen = collections.deque(seed, self.order)
                        self.seen.extend(seed)
                        output += ' '.join(seed) + ' '
                        # Leave the loop and proceed to prepare the output
                        break
            else:
                # If the seed is emptied because of elimination (or it was empty already)
                # Start creating new one by choosing a random word from the table
                output = "Sorry, I dont know the answer.\n\nBut there is another thing to think about;\n\n"
                seed = list(random.choice(list(self.table.keys())))

                # for ms in self.word_vectors.most_similar(seed[0]):
                #     if ms in list(self.table.keys()):
                #         seed = ms
                #         break

        # Prepare the output based on self.seen
        for i in range(max_words):
            word = random.choice(self.table[tuple(self.seen)])
            if word == NONWORD:
                exit()
            if newline_after is not None and i % newline_after == 0:
                output += word + '\n'
            else:
                output += word + ' '
            self.seen.append(word)
        return output

    def walk_directory(self, rootDir):
        for dirName, subdirList, fileList in os.walk(rootDir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                self.generate_table(os.path.join(dirName, fname))
