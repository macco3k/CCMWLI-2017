Simge Ekiz(s4706757) - Katrin Bujari(s1005213) - Daniele Maccari (...)

# Some preliminary ideas

For a start-up, we had different questions to answer:
* What training data?
* What topic?
* What model?

In the end, since we are generating a chatbot, we settled for a dump from the philosophy section of the stackexchange network (files can be found at https://archive.org/details/stackexchange). This way, even if the chatbot start blabbering nonsense, we can excuse him, as he's a philosopher after all :D
With respect to the mode, we noted that while markov n-grams look nice for generating text, it's not really clear how to use them to actually answer questions. One intuitive solution would be to train one network per tag, or category, so that different networks could generate different texts based on the topic chosen by the user. Other, more complex models, e.g. seq2seq (https://github.com/farizrahman4u/seq2seq) and pattern-based approaches (https://en.wikipedia.org/wiki/AIML) were dropped for lack of time, though they would be very interesting to explore, in particular in combination.

Ideally, a conversation would look like the following:

`"Hi, My name is <botname>. I can do this and this and also that. To see a list of commands type /help."`

`"Hello, I'm Daniele. Let's talk about existentialism"`

_...the bot look up existensialism in a db of tags..._

`"That seems to be a pretty big topic. Can you be more specific?"`

`"Something about Kierkegaard's vision of life?"`

Of course, the major problem here is to find a way for the bot to understand the question.

For both problems, i.e. clustering questions by topic and understanding the user question, we need some metric to measure documents' similarity, so that we can e.g. find the main topic of a user's question and then pick the right bot for the topic. To define such a metric, we took a couple of approaches, the first based on word2vec (or glove) vectors, and the second on LDA. In the end, we decided to use LDA, as word2vec similarities proved to be unusable, probably also due to the high specificity of the subject at hand. For more details, see also the `process_and_train` notebook, which contains more extensive comments.

# Implementation details
For the final implementation, we had the following pipeline:

1. Perform data preprocessing. This includes standard nlp techniques such as stemming, stopwords and punctuation removal, etc. The final result is saved in a csv file for later use.
1. Associate each question which the 'best' answer, be it the accepted one or the answer with the highest vote-count. In order to do this we joined each question by using pandas `merge` function, thus getting a refined dataframe.
1. Process the resulting output in order to build a corpus of bow-documents usable for the LDA model training.
1. Generate the LDA model with a fixed number of topics (in our case, 10).
1. Based on such model, categorize each question into one topic.
1. Create a text file per topic, containing all the answers to questions belonging to that topic according to the model.
1. Train a different markov network on each of these files. Each network will act as the topic's bot.
1. At runtime, parse the user's question in the same way as we did for the documents, and extract its main topic.
1. Pick the bot responsible for that topic and generate some reply.

# Comments
We tried different versions of the networks, using the base implementation we were provided with and a more sophisticated library which we slightly adapted to our case (the original can be found at https://github.com/pteichman/cobe).

Because of the way the inference process is implemented in the base version, there's no way to provide the network with a 'seed'. That is, to provide some context to boot it up with an n-gram from the query which would give the answer a bit of context. One solution we came up with was to generate different order networks, from 1 up to `order`, where order is the maximum order specified as a parameter of the network. As an initial seed, we picked part of the user's query If that could not be found, we fell back to lower order network, down to order 1
Our chatbot takes an input from the user something 'what is life' and it searches the corpus then if it doesn't found in the corpus , it removes one word that has lower tf-idf score. for this example let's assume 'what' is dropped. Now it searches 'is life' in the corpus if it doesn't found again it drops one more word according to tf-idf score. This continue until the word is found in the corpus. If word doesn't found in the corpus we select random word. After we found some word or word sequence that exist in the corpus we start to increase the markov order to the some level that is specified before in our case it is 7. That is, we selected the higher order network to select the next word to output. The reasoning behind this being that, by moving to lower-order networks, we have a higher change of finding the seed, with the extreme case being that of selecting just a single word. From there, going back to higher-order networks means we try to recover some context by looking for longer n-grams. Unfortunately, this requires a lot more computation, as at runtime, all these networks needs to be either generated on-the-fly or at least queried (and also in the latter case, we need to generate the network at initialization time).

Examples:

![Screen1](https://github.com/macco3k/CCMWLI-2017/blob/master/wg4/screenshots/1.png)
![Screen2](https://github.com/macco3k/CCMWLI-2017/blob/master/wg4/screenshots/2.png)

A second option was to rely on an external library (cobe), which already implemented a similar mechanism. The library builds both a forward and a backward versions of the same network. At query time, the network is traversed to find the best answer given some time-limit the user can provide (in our experiments, we set it to a couple of seconds). This traversal makes use of the user's query by selecting some pivot-words to seed the search. A number of answers are generated, with the best one being picked, according to a scoring system.

![Screen3](https://github.com/macco3k/CCMWLI-2017/blob/master/wg4/screenshots/3.png)
![Screen4](https://github.com/macco3k/CCMWLI-2017/blob/master/wg4/screenshots/5.png)

Performance in both cases was pretty low, though in the latter some more intelligible text could be produced, due to the smarter algorithm used by the library.

As it is easy to imagine, the bot is in both cases quite limited, in that the only realy clue it has about the content of the question is the query's topic. In particular, even if the coherence (i.e. the inter-similarity) of questions the lda model was trained on were high enough to get a meaningful clustering, ten topics are probably not enough to really capture the full set of themes one could talk about in philosophy. This is especially true when dealing with short sentences typical of a chat. Also, we did not use any of the information about the tags associated with the questions, despite intuitively they already provide some sort of ground truth for the theme the question covers. On the other hand, multiple tags are usually linked to each question, sometime also very different, with no specific hierarchy defined among them. On the answer side, we note how this very simplistic approach trains a network on a very big answer-file, which containes every answer pertaining with a topic. Of course, this makes the variance withing the document much higher than in single answers which can be found on the website. To tackle this problem, questions with very similar topic composition could be clustered together. A sub-topic lookup table could then be constructed so that only answers for very similar questions (e.g. near duplicates) would end up in the same corpus (and thus used to train the same network). Moreover, we did not inspect the query for particular categories (e.g. pos-tags) nor entities which could be useful to extract information about the type of question (i.e. w- questions), which could in turn be used to seed the network. Finally, the bot has of course no notion of context, as it simply generates random continuations of sentences based on the training data it's been exposed to. In this respect, models including such "memory" aspect, e.g. lstms, could prove beneficial.
