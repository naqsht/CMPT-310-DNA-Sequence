#!/usr/bin/python3

import sys
import os
import random
import math

import numpy as np
import operator

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 0
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
# <Your feedback goes here>
#####################################################
#####################################################



# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0


class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior      = np.array([0.5, 0.5])
        self.transition = np.array([[0.999, 0.001], [0.01, 0.99]])
        self.emission   = np.array([{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                                    {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}])

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def viterbi(self, sequence):
        ###########################################
        # Start your code
        print("My code here")
        T = len(sequence)
        D = self.num_states

        # Initialize matrices
        m = np.zeros((T,D))
        prev = np.zeros((T,D))
        
        m[0,:] = self.prior
        
        # Viterbi algorithm loops

        for t in range(1,T):
            for i in range(D):
                prob = [None]*D
                for j in range(D):
                    prob[j] = (m[t-1,j])*(self.transition[j][i])*(self.emission[i][sequence[t]])
                m[t,i] = max(prob)
                prev[t,i] = np.argmax(prob)
        
        
        path = [int(0)]*T
        path[T-1] = int(np.argmax(m[T-1,:]))

        # Constructing the path
        for t in range(T-2,-1,-1):
            x = int(path[t+1])
            path[t] = int(prev[t+1,x])
        
        return path
        # End your code
        ###########################################


    def log_sum(self, factors):
        if abs(min(factors)) > abs(max(factors)):
            a = min(factors)
        else:
            a = max(factors)

        total = 0
        for x in factors:
            total += math.exp(x - a)
        return a + math.log(total)

    # - sequence: String with characters [A,C,T,G]
    # return: posterior distribution. shape should be (len(sequence), 2)
    # Please use log_sum() in posterior computations.
    def posterior(self, sequence):
        ###########################################
        # Start your code
        print("My code here")
        T = len(sequence)
        D = self.num_states
        
        f = np.zeros((T,D))
        b = np.ones((T,D))
        
        # Forward algorithm
        for i in range(D):
            f[0][i] = np.log(self.prior[i]) + np.log(self.emission[i][sequence[0]])
        

        for t in range(1,T):
            for i in range(D):
                temp_listf = [0]*D
                for j in range(D):
                    temp_listf[j] = f[t-1][j] + np.log(self.transition[j][i]) + np.log(self.emission[i][sequence[t]])
                f[t][i] = self.log_sum([temp_listf[0],temp_listf[1]])
        

        # Backward algorithm
        b[T-1,:] = 0

        for t in range(T-2,-1,-1):
            for i in range(D):
                temp_listb = [0]*D
                for j in range(D):
                    temp_listb[j] = b[t+1][j] + np.log(self.transition[i][j]) + np.log(self.emission[j][sequence[t+1]])
                b[t][i] = self.log_sum([temp_listb[0],temp_listb[1]])
        
        

        alpha = 1/self.log_sum(f[:,-1])
        
        # Combining forward and backward
        post_prob = np.zeros((T,D))
        

        for i in range(T):
            for j in range(D):
                post_prob[i][j] = (f[i][j])*(b[i][j])*(alpha)
        

        return post_prob


        # End your code
        ###########################################


    # Output the most likely state for each symbol in an emmision sequence
    # - sequence: posterior probabilities received from posterior()
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def posterior_decode(self, sequence):
        nSamples  = len(sequence)
        post = self.posterior(sequence)
        best_path = np.zeros(nSamples)
        for t in range(nSamples):
            best_path[t], _ = max(enumerate(post[t]), key=operator.itemgetter(1))
        return list(best_path.astype(int))


def read_sequences(filename):
    inputs = []
    with open(filename, "r") as f:
        for line in f:
            inputs.append(line.strip())
    return inputs

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, viterbi, posterior):
    vit_file_name = filename[:-4]+'_viterbi_output.txt' 
    with open(vit_file_name, "a") as f:
        for state in range(2):
            f.write(str(viterbi.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, viterbi)))
        f.write("\n")

    pos_file_name = filename[:-4]+'_posteri_output.txt' 
    with open(pos_file_name, "a") as f:
        for state in range(2):
            f.write(str(posterior.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, posterior)))
        f.write("\n")


def truncate_files(filename):
    vit_file_name = file[:-4]+'_viterbi_output.txt'
    pos_file_name = file[:-4]+'_posteri_output.txt' 
    if os.path.isfile(vit_file_name):
        open(vit_file_name, 'w')
    if os.path.isfile(pos_file_name):
        open(pos_file_name, 'w')


if __name__ == '__main__':

    hmm = HMM()

    file = sys.argv[1]
    truncate_files(file)
    
    sequences  = read_sequences(file)
    for sequence in sequences:
        viterbi   = hmm.viterbi(sequence)
        posterior = hmm.posterior_decode(sequence)
        write_output(file, viterbi, posterior)


