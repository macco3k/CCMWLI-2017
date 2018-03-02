#!/usr/bin/env python
# GloVe trained on Wikipedia
# http://nlp.stanford.edu/data/glove.6B.zip

# re-save pretrained GloVe in word2vec format
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

## Re-save pretrained GloVe in word2vec format
#glove_input_file = 'glove.6B.50d.txt'
#word2vec_output_file = 'glove.6B.100d.txt.word2vec'
#glove2word2vec(glove_input_file, word2vec_output_file)
#
## Load the Stanford GloVe model
#filename = 'glove.6B.100d.txt.word2vec'
#model = KeyedVectors.load_word2vec_format(filename, binary=False)


# Load Google's pre-trained Word2Vec model
#model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
# has plural and upper/lower case, and even bigrams (e.g., taxpayer_dollars; vast_sums)

## flex word2vec's muscles
#model.doesnt_match("man woman child kitchen".split())
#model.doesnt_match("france england germany berlin".split())
#model.doesnt_match("paris berlin london austria".split())
#model.most_similar("amsterdam")
#
## Consider a two-person task with a signaler and a receiver (similar to the TV gameshow 'Password'):
## The signalers were told that they would be playing a word-guessing game in which 
## they would have to think of one-word signals that would help someone guess their items. 
## They were talked through an example: if the item was 'dog', then a good signal would be 
## 'puppy' since most people given 'puppy' would probably guess 'dog'.
#
## sender thinks bank, says money
## receiver think cash
#model.most_similar("bank") # .69 robber, .67 robbery, robbers, security, agency ..
#model.most_similar("money") # .55 dollars, .55 profit, .54 cash
#model.most_similar("cash") # .69 capitalize, .54 money, sell, debt, tax

lemmatizer = WordNetLemmatizer()                   
stemmer = PorterStemmer()                   
                   
## TODO: include a number for the function to check for not 10,but 20 poss. suggestions etc if empty
#def returnValidPossibilities(word):
#    # First of all, extract all suggestions that immediately pop up according to word2vec:
#    suggestions = model.most_similar(word)
#    valid_suggestions = []
#    # Extract stemmed version of the input word:
#    #lemmatized_word = lemmatizer.lemmatize(word.lower())    
#    lemmatized_word = stemmer.stem(word.lower())
#    for index,item in enumerate(suggestions): 
#        # Extract stemmed version of password:
#        #lemmatized_suggestion = lemmatizer.lemmatize(item[0].lower())
#        lemmatized_suggestion = stemmer.stem(item[0].lower())
#        # Now, check whether the actual word is contained in a possible suggestion
#        # If yes, ignore suggestion; else append suggestion to valid ones
#        if(lemmatized_word not in lemmatized_suggestion):
#            # Check whether the lemmatized version is already inside:
#            if(lemmatized_suggestion not in dict(valid_suggestions)):
#                # If it is not already inside our valid suggestions, append it
#                valid_suggestions.append((lemmatized_suggestion, item[1]))
#    # Check whether the suggestion itself is part of the vocabulary            
#    # 2nd step: only keeping lemmatized version of the same object:
#        
#    return valid_suggestions    

                  
                  
# TODO: include a number for the function to check for not 10,but 20 poss. suggestions etc if empty
def returnValidPossibilities(word):
    # First of all, extract all suggestions that immediately pop up according to word2vec:
    suggestions = model.most_similar(word)
    # Initialize list for storing valid suggestions as tuples
    valid_suggestions = []
    # Create the following variable to keep track of what we already stored:
    lemmatized_valid_suggestions = []    
    
    # Extract stemmed version of the input word:
    #lemmatized_word = lemmatizer.lemmatize(word.lower())    
    lemmatized_word = stemmer.stem(word.lower())
    for index,item in enumerate(suggestions): 
        # Extract stemmed version of password:
        #lemmatized_suggestion = lemmatizer.lemmatize(item[0].lower())
        lemmatized_suggestion = stemmer.stem(item[0].lower())
        # Now, check whether the actual word is contained in a possible suggestion
        # If yes, ignore suggestion; else append suggestion to valid ones
        if(lemmatized_word not in lemmatized_suggestion):
            # Check whether the lemmatized version is already inside:
            if(lemmatized_suggestion not in lemmatized_valid_suggestions):#dict(valid_suggestions)):
                # If it is not already inside our valid suggestions, append it
                valid_suggestions.append(item)
                # Store the lemmatized version so that it can be checked later:
                lemmatized_valid_suggestions.append(lemmatized_suggestion)    
    # Check whether the suggestion itself is part of the vocabulary            
    # 2nd step: only keeping lemmatized version of the same object:
        
    return valid_suggestions                    
                  

def send_word(secret):
    # TODO:
    # does the word itself even appear in suggestions of word to send?
    #     
    
    word_to_send = 'Call me maybe'    
    return word_to_send    





                  
#model['money']
#
#model.similarity("hot","cold") # .20
#model.similarity("hot","warm") # .14

                
