import sys, random, collections, os, re
from nltk import word_tokenize

# Since we split on whitespace, this can never be a word
NONWORD = "\n"

class Markov():
    """
    This basically parse the text through a fixed-size sliding window (the order param).
    A dict stores a mapping between n-grams of size=order and the word coming right next.
    At generation time, the dictionary is queried for possible words in the following loop:
        - first, the most probable next word is chosen given the last seen n-gram
        - second, this word is added to the list of seen word to generate the next n-gram.
          This is equivalent to a one-step slide of the window.
        - the loop is repeated
    """
    def __init__(self, order=2):
        self.order = order
        self.table = collections.defaultdict(list)
        self.seen = collections.deque([NONWORD] * self.order, self.order)

    # Generate table
    def generate_table(self, filename):
        for line in open(filename):
            # for word in re.split(r' ', line):
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

        if seed:
            self.seen = collections.deque(seed, self.order)
            print(self.seen)
            # self.seen.extend(seed)
            try:
                word = random.choice(self.table[tuple(self.seen)])
                print(self.table[tuple(self.seen)])
            except IndexError as e:
                print("Seed is not found in the corpus.")
                self.seen.extend([NONWORD] * self.order)
        else:
            self.seen.extend([NONWORD] * self.order)  # clear it all

        for i in range(max_words):
            word = random.choice(self.table[tuple(self.seen)])
            #print(word)
            if word == NONWORD:
                exit()
            if i % newline_after == 0:
                output += word + '\n'
            else:
                output += word + ' '
            self.seen.append(word)
        #print('Output:', output)
        return output

    def walk_directory(self, rootDir):
        for dirName, subdirList, fileList in os.walk(rootDir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                self.generate_table(os.path.join(dirName, fname))
            # print('\t%s' % fname)


m = Markov(order=3)

# m.walk_directory('./pres-speech')
#m.walk_directory('./pres-speech/clinton')
#m.generate_output(max_words=50, seed=['they','thank','of'])
# TODO crete different order network
# aiml
