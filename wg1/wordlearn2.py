import numpy as np


# each row is a trial, comprised of a subset of word (object) indices 
# sampled from the entire vocabulary
hiCDord = np.genfromtxt('freq369-3x3hiCD.txt', delimiter='\t')
loCDord = np.genfromtxt('freq369-3x3loCD.txt', delimiter='\t')


# define a function that accepts an array of trials (e.g., hiCDord)
# and returns a word x object co-occurrence matrix
def coocMatrix(ord):
	# nw = number of words
	# no = number of objects
	M = np.zeros(shape=(nw,no)) 
	# counting...
	return M


# define a test function that accepts a 'memory' matrix
# --containing word-object hypotheses or associations--
# and a decision parameter, and returns choice probabilities
# of each object, given each word, according to softmax:
# https://en.wikipedia.org/wiki/Softmax_function
# (decision parameter = RL 'temperature')
def softmax(M,temp):
	return prob_correct

# define a function that accepts an array of trials and parameter
# values, and returns a memory matrix with the learned representation
def model(ord, par):
	M = np.zeros(shape=(nw,no)) 
	# learning: i.e., not just co-occurrence counting,
	# but a process that corresponds to what you think 
	# people might be doing as they go through the trials
	# (guess-and-test hypothesis generation? biased association?)
	return M # this matrix will then be passed through softmax to extract pr(correct)

# define a function that accepts a vocabulary (1:M), a 
# distribution over the likelihood of sampling each word (object)
# and a desired number and size of trials, and returns a trial order
# (e.g., to accomplish simulations like those in Blythe et al., (2016))


# graph the mean performance for different softmax parameter values (e.g., .1 to 10)
# http://matplotlib.org/users/pyplot_tutorial.html
# you can first try feeding a co-occurrence matrix through softmax, 
# and then try your cognitive model's output
import matplotlib.pyplot as plt
def plot_performance_by_temperature(ord):
	temp = np.arange(.1,10.2,.5)
	meanPerf = np.zeros(len(temp))
	# for each temp, call softMax(coocMatrix(ord)) and save the mean perforfmance
	plt.plot(temp, meanPerf)


### Evaluating model fit ###

# try implementing each of the following three methods (SSE, crossEntropy, 
# and negative log likelihood) and get a sense of their values for varying discrepancies of p and q
# human response probabilities for each correct of the 18 correct pairs:
# human_accuracy_variedCD.csv has columns hiCDacc and loCDacc

# implement sum of squared error measure of model fit (p) to observed data (q)
def SSE(p,q):
	return SSE

# implement cross entropy measure
def crossEntropy(observed_probs,model_probs):
	return xent
  

# implement negative log-likelihood measure, assuming each test 
# problem is binomial (since I didn't give you the full response matrix)
def negloglik(obs, mod):
	return nll

# implement a function (BIC) calculating the Bayesian Information Criterion
# https://en.wikipedia.org/wiki/Bayesian_information_criterion



### Fitting a model ### 

# given a trial-ordering, model parameters, and a set of human Pr(correct),  
# return your favorite goodness-of-fit value
def evaluateModel(parms, order, hum):
	return fitval


# write a function for optimizing parameter values (for a given trial-ordering
# and human Pr(correct)
# https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/optimize.html

