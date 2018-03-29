import sys, random, collections, os, re
from nltk import word_tokenize

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
    def __init__(self, topic, order=1):
        self.order = order
        self.topic = topic
        self.table = collections.defaultdict(list)
        self.seen = collections.deque([NONWORD] * self.order, self.order)

    # Generate table
    def generate_table(self, filename):
        for line in open(filename, 'r', encoding='utf-8'):
            for word in word_tokenize(line):
                if word != NONWORD:
                    word = word.lower()
                    self.table[tuple(self.seen)].append(word)
                    self.seen.append(word)
        self.table[tuple(self.seen)].append(NONWORD)  # Mark the end of the file

    # table, seen = generate_table("gk_papers.txt")

    # Generate output
    def generate_output(self, max_words=100, newline_after=10, seed=None):
        output = ''

        self.update_order(1)
        if (seed):
            seed = [seed[0]]
        else:
            seed = [NONWORD]
        for i in range(max_words):

            if self.order < 5:

                if i is not self.order-1:
                    print("updating order to", i+1)
                    self.update_order(i+1)

                if seed:
                    self.seen = collections.deque(seed, self.order)
                    # self.seen.extend(seed)
                    try:
                        word = random.choice(self.table[tuple(self.seen)])
                    except IndexError as e:
                        print("Seed is not found in the corpus.")
                        self.seen.extend(random.choice(list(self.table.keys())))
                        seed = []
                        # self.seen.extend([NONWORD] * self.order)
                else:
                    self.seen.extend([NONWORD] * self.order)  # clear it all

            word = random.choice(self.table[tuple(self.seen)])
            if word == NONWORD:
                exit()
            if newline_after is not None and i % newline_after == 0:
                output += word + '\n'
            else:
                output += word + ' '
            self.seen.append(word)

            if self.order < 5:
                seed.append(word)
        #print('Output:', output)
        return output

    def walk_directory(self, rootDir):
        for dirName, subdirList, fileList in os.walk(rootDir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                self.generate_table(os.path.join(dirName, fname))
            # print('\t%s' % fname)

    def update_order(self, new_order):
        self.order = new_order
        self.seen = collections.deque([NONWORD] * self.order, self.order)
        self.generate_table(self.topic)


# m = Markov(order=3)

# m.walk_directory('./pres-speech')
#m.walk_directory('./pres-speech/clinton')
#m.generate_output(max_words=50, seed=['they','thank','of'])
# TODO crete different order network
# aiml
