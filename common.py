from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1
from math import sqrt
from PIL import Image
import os
import tensorflow as tf
import pickle
from itertools import product

def generateParameters(elongation=4, badcellperc=0.3):
    r = 4
    
    eps2 = elongation
    p1 = badcellperc
    p2 = 1 - p1
    
    a1 = r
    b1 = r
    
    a2 = (eps2 ** 0.5) * r
    b2 = r / (eps2 ** 0.5)
    
    return [p1, a1, b1, p2, a2, b2]

class DiffractionPatternGenerator:
    def __init__(self, 
                 z=65_000,
                 lmbda=0.63,
                 max_angle_degree=15,
                 n_points=1000):
        self.z = tf.Variable(z, dtype=float)
        self.lmbda = tf.Variable(lmbda, dtype=float)
        self.k = tf.Variable(2 *np.pi/lmbda, dtype=float)
        
        max_angle_degree = 15;
        max_angle = max_angle_degree * np.pi / 180;

        x_max = z * np.tan(max_angle)
        y_max = z * np.tan(max_angle)

        n_points = 1000 # resolution of our diffraction pattern picture in both axis
        x_ar = np.linspace(-x_max, x_max, n_points)
        y_ar = np.linspace(-y_max, y_max, n_points)
        
        buf = np.meshgrid(y_ar[:y_ar.shape[0]], x_ar[:x_ar.shape[0]])
        self.x_data = tf.Variable(buf[1], dtype=float)
        self.y_data = tf.Variable(buf[0], dtype=float)
        
        self.p1 = tf.Variable(0, dtype=float)
        self.a1 = tf.Variable(0, dtype=float)
        self.b1 = tf.Variable(0, dtype=float)
        self.p2 = tf.Variable(0, dtype=float)
        self.a2 = tf.Variable(0, dtype=float)
        self.b2 = tf.Variable(0, dtype=float)
        self.q1 = tf.Variable(0,shape=tf.TensorShape(None), dtype=float)
        self.q2 = tf.Variable(0,shape=tf.TensorShape(None), dtype=float)
        
    def I_tf(self):
        self.q1.assign(self.k/self.z*tf.sqrt(tf.square(self.a1 * self.x_data) + tf.square(self.b1 * self.y_data)))
        self.q2.assign(self.k/self.z*tf.sqrt(tf.square(self.a2 * self.x_data) + tf.square(self.b2 * self.y_data)))
        
        return self.p1 * tf.square(self.a1 * self.b1 * tf.math.special.bessel_j1(self.q1) / self.q1) +\
               self.p2 * tf.square(self.a2 * self.b2 * tf.math.special.bessel_j1(self.q2) / self.q2)
    
    def getPattern(self, params):
        self.p1.assign(params[0])
        self.a1.assign(params[1])
        self.b1.assign(params[2])
        self.p2.assign(params[3])
        self.a2.assign(params[4])
        self.b2.assign(params[5])
        
        return self.I_tf()
    
class DiffMCMCSampler():
    
    def __init__(self):
        self.generator = DiffractionPatternGenerator()
        self.sigma = tf.Variable(0, dtype=float)
        self.pi2 = tf.constant(2 * np.pi, dtype=float)
    
    def transition_model(self, params, it, lr):
        p1, a1, b1, a2, b2, sigma = params
        scale = lr# * 10 ** -(it // 1000)
        
        loc = np.array([p1, a1, b1, a2, b2, sigma])
        noise = np.random.normal(loc=0, scale=scale, size=loc.shape[0])

        return loc+noise
    
    def prior(self, params):
        p1, a1, b1, a2, b2, sigma = params
        if  sigma > 0 \
        and a1 > 0 \
        and b1 > 0 \
        and a2 > 0 \
        and b2 > 0 \
        and a1 >= b1 \
        and a2 >= b2 \
        and 0 < p1 < 1\
        and  0.95 < a1 / b1 < 1.05\
        and 0.95 < (a2 * b2) / (a1 * b1) < 1.05:
            return True
        return False
    
    def loglike(self, params):
        p1, a1, b1, a2, b2, sigma = params
        params_tf = [p1, a1, b1, 1-p1, a2, b2]
        self.sigma.assign(sigma)
        lp = self.prior(params)
        if not lp:
            return None
        else:
            return -tf.reduce_sum(tf.math.squared_difference(self.I_true,self.generator.getPattern(params_tf) )) / (tf.square(self.sigma) * 2)-\
                    self.size*tf.math.log(self.sigma*tf.sqrt(self.pi2))

    def acceptance(self, params, params_new):
        if params_new is None:
            return False
        
        if params_new > params:
            return True
        else:
            accept=np.random.uniform(0,1)
            return (accept < (np.exp(params_new-params)))
        
    def run_mcmc(self, param_init, num_iterations, true_data, lr):
        params = param_init
        
        burn_in = int(num_iterations * 0.4)
        self.I_true = tf.Variable(true_data, dtype=float)
        self.size = tf.cast(tf.size(self.I_true), float)
        accepted = []
        all_params = np.empty([num_iterations, len(params)])
        states = np.empty([num_iterations, len(params)])
        
        params_lik = self.loglike(params)

        for i in range(num_iterations):
            states[i] = params
            params_new =  self.transition_model(params, i, lr)    
            params_new_lik = self.loglike(params_new)
            if (self.acceptance(params_lik,params_new_lik)):            
                params = params_new
                params_lik = params_new_lik
                if (i >= burn_in):
                    accepted.append(params_new)  
            all_params[i] = params_new
        return np.array(accepted), np.array(states), np.array(all_params)