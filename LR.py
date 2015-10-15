# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:53:59 2015

@author: congxiu
"""
import numpy as np

class LogisticRegression:
    def __init__(self, sample, label):
        self.sample = sample
        self.label = label
        self.w = np.array([0] * sample.shape[1], ndmin = 2).T
        self.size = sample.shape[0]
    
    @staticmethod    
    def logit(z):
        return 1 / (1 + np.exp(-z))
        
    def gradient(self, w):
        G = (self.logit(-self.label * self.sample.dot(w)) * (-self.sample * self.label)).sum(axis = 0)
        G = G / self.size
        G.shape = (G.shape[0], 1)
        
        return G
        
    def stochasticGradient(self, w, n):
        G = self.logit(-self.label[n] * self.sample[n].dot(w)) * (-self.label[n] * self.sample[n])
        G.shape = (G.shape[0], 1)
        
        return G
    
    def train(self, eta = 0.01, iteration = 2000, stochastic = False):
        w = self.w
        for i in range(iteration):
            if stochastic:
                w = w - eta * self.stochasticGradient(w, i % self.size)
            else:
                w = w - eta * self.gradient(w)
            
        self.w = w
        
        print "Traning accuracy is", self.score(self.sample, self.label)
        
    def predict(self, sample, threshold = 0.5):
        p = self.logit(sample.dot(self.w))
        result = p
        result[p >= threshold] = 1
        result[p < threshold] = -1
        
        return result
        
    def score(self, sample, label):
        return (self.predict(sample) == label).mean()