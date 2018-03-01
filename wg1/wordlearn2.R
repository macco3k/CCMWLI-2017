root_path = 'D:\\OneDrive\\Documenti\\Radboud\\2017\\Semester 2\\CCMLWI\\Assignments\\wg1'
# each row is a trial, comprised of a subset of word (object) indices 
# sampled from the entire vocabulary

# Experiment 3 in the paper.
# Total trials = 36. Total pairs = 18. 
# Each trial is composed of 3 pairs. 6 pairs appear 3 times,
# another 6 pairs appear 6 times, and the last 6 pairs appear 9 times.

# hiCD pairs were allowed to mix together with no restriction
hiCDord = read.table(file.path(root_path, 'freq369-3x3hiCD.txt'), sep='\t', header=F)
# lowCD pairs were only allowed to mix with their own subsgroup (3-3, 6-6, 9-9)
loCDord = read.table(file.path(root_path, 'freq369-3x3loCD.txt'), sep='\t', header=F)

# define a function that accepts an array of trials (e.g., hiCDord)
# and returns a word x object co-occurrence matrix
coocMatrix <- function(ord) {
  N = max(unlist(ord)) # the max pairs we see is the # of pairs
  M = matrix(0, nrow=N, ncol=N)
  for(t in 1:nrow(ord)) { # for each trial
    tr = unlist(ord[t,])  # count which word-object pair occurred in the trial. Pair each word with each object, as we don't know which word relates to which object
    M[tr,tr] = M[tr,tr] + 1
  }
  return(M)
}

# define a test function that accepts a 'memory' matrix
# --containing word-object hypotheses or associations--
# and a decision parameter, and returns choice probabilities
# of each object, given each word, according to softmax:
# https://en.wikipedia.org/wiki/Softmax_function 
# (decision parameter = RL 'temperature')
softmax <- function(M, temp) {
  expM = exp(M/temp)
  return( diag(expM) / rowSums(expM) )
}

ent <- function(p) {
  H = -(p %*% log(p))
}


# define a function that accepts an array of trials and parameter
# values, and returns a memory matrix with the learned representation

model <- function(ord, parms) {
  # nw = number of words
  # no = number of objects
  # M = matrix(0, nrow=parms$nw, ncol=parms$n)
  # learning: i.e., not just co-occurrence counting,
  # but a process that corresponds to what you think 
  # people might be doing as they go through the trials
  # (guess-and-test hypothesis generation? biased association?)
  
  # implement the exploitation-exploration model of 1_Kachergis_Yu_Shiffrin2012_pbr
  alpha = parms[[1]]  # controls the forgetting rate
  chi = parms[[2]]    # weight (attention) to be distributed among new and familiar pairings
  lambda = parms[[3]] # scaling parameters governing how much to weigh uncertainty
  
  for(t in 2:nrow(ord)) {
    M = coocMatrix(as.matrix(ord[1:t,]))
    prob_M = M/rowSums(M)
    
    # prevent NaNs
    prob_M[prob_M == 0] = 1e-2
    
    # compute entropies over words(rows) and objects(cols) (though the matrix is simmetric, so it doesn't matter?)
    H_w = apply(prob_M,1,ent)
    H_o = apply(prob_M,2,ent)
    
    exp_M = exp(lambda*(H_w+H_o)) * M
    M = alpha*M + (chi * exp_M)/sum(exp_M)
    
  }
  
  return(M) # this matrix will then be passed through softmax to extract pr(correct)
}

# define a function that accepts a vocabulary (1:M), a 
# distribution over the likelihood of sampling each word (object)
# and a desired number and size of trials, and returns a trial order
# (e.g., to accomplish simulations like those in Blythe et al., (2016))
genTrialsOrder <- function(vocab, prob, ntrials, size) {
  trials = matrix(0, nrow=ntrials, ncol=size)
  for (t in 1:ntrials) {
    trials[t,] = sample(vocab, size, replace=FALSE, prob=prob)
  }
  
  return (trials)
}

# graph the mean performance for different softmax parameter values (e.g., .1 to 10)
require(ggplot2)
plot_performance_by_temperature <- function(ord) {
  temp = seq(.1,10,.5)
  item_perf = matrix(0, nrow=length(temp), ncol=18)
  for(i in 1:length(temp)) {
    item_perf[i,] = softmax(coocMatrix(ord), temp[i])
  }
  dat = data.frame(cbind(temp, perf = rowMeans(item_perf)))
  ggplot(dat, aes(temp, perf)) + geom_point(color="red", alpha=.5) + geom_line(alpha=.5)
}

### Evaluating model fit ###
# try implementing each of the following three methods (SSE, crossEntropy, 
# and negative log likelihood) and get a sense of their values for varying discrepancies of p and q
# human response probabilities for each correct of the 18 correct pairs:
load("human_accuracy_variedCD.RData") # hiCDacc and loCDacc

# implement sum of squared error measure of model fit (p) to observed data (q)
SSE <- function(p, q) {
  sse = (p-q) %*% (p-q)
  return(as.numeric(sse))
}

# implement cross entropy measure
crossEntropy <- function(observed_probs, model_probs) {
  xent = -((observed_probs) %*% log(model_probs))
  return(as.numeric(xent))
}

# implement negative log-likelihood measure, assuming each test 
# problem is binomial (since I didn't give you the full response matrix)

# we assume obs is the binary response (paired/unpaired), while p contains the model estimate
# for the probability of association of each <word-object> pair
negloglik <- function(obs, p) {
  pointwise_ll = obs*log(p) + (1-obs)*log(1-p)
  pointwise_ll[is.na(pointwise_ll)] = 0 # prevent 0 * -Inf when probabilities goes to 1
  nll = -sum(pointwise_ll)
  return(as.numeric(nll))
}

# implement a function (BIC) calculating the Bayesian Information Criterion
# https://en.wikipedia.org/wiki/Bayesian_information_criterion
bic <- function(ll, n, k) {
  bic = log(n)*k - 2*ll
}


### Fitting a model ### 

temp = 1e-1
# given a trial-ordering, model parameters, and a set of human Pr(correct),  
# return your favorite goodness-of-fit value.
# How can we compute the likelihood of the human observations?
evaluateModel <- function(parms, order=loCDord, hum=loCDacc) {
  mod = model(order, parms)
  p = softmax(mod, temp)
  
  writeLines('Association matrix:\n')
  print(mod)
  
  writeLines('\nProbabilities:\n')
  print(p)
  
  n = length(hum)
  k = length(parms)
  
  # observations are just 18 ones?
  obs = matrix(1, ncol=n)
  #nll = negloglik(obs, p)
  
  fitval = SSE(hum, p)
  return(fitval)
}

require(DEoptim)
optim = DEoptim(evaluateModel, c(0, 0, 0), c(1, 50, 10))
