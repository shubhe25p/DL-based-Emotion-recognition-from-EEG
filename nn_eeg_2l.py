# Copyright (c) 2020 Shubh Pachchigar
"""
EEG Data is taken from DEAP
The training data was taken from DEAP.
See my:
- Github profile: https://github.com/shubhe25p
- Email: shubhpachchigar@gmail.com
"""


import numpy as np
import h5py
import scipy.spatial as ss
import scipy.stats as sst
import _pickle as cPickle
from pathlib import Path
import itertools
import random

def get_frequency(all_channel_data): 
	L = len(all_channel_data[0])
	Fs = 128
	data_fft = map(lambda x: np.fft.fft(x),all_channel_data)
	frequency = map(lambda x: abs(x//L),data_fft)
	frequency = map(lambda x: x[: L//2+1]*2,frequency)
	f1,f2,f3,f4,f5=itertools.tee(frequency,5)
	delta = np.array(list(map(lambda x: x[L*1//Fs-1: L*4//Fs],f1)))
	theta = np.array(list(map(lambda x: x[L*4//Fs-1: L*8//Fs],f2)))
	alpha = np.array(list(map(lambda x: x[L*5//Fs-1: L*13//Fs],f3)))
	beta =  np.array(list(map(lambda x: x[L*13//Fs-1: L*30//Fs],f4)))
	gamma = np.array(list(map(lambda x: x[L*30//Fs-1: L*50//Fs],f5)))

	return delta,theta,alpha,beta,gamma
	
def get_feature(all_channel_data): 
	
	(delta,theta,alpha,beta,gamma) = get_frequency(all_channel_data)
	delta_std = np.std(delta,axis=1)
	theta_std = np.std(theta,axis=1)
	alpha_std = np.std(alpha,axis=1)
	beta_std = np.std(beta,axis=1)
	gamma_std = np.std(gamma,axis=1)
	delta_mean = np.mean(delta,axis=1)
	theta_mean = np.mean(theta,axis=1)
	alpha_mean = np.mean(alpha,axis=1)
	beta_mean = np.mean(beta,axis=1)
	gamma_mean = np.mean(gamma,axis=1)
	feature = np.array([delta_std,theta_std,alpha_std,beta_std,gamma_std,delta_mean,theta_mean,alpha_mean,beta_mean,gamma_mean])
	feature = feature.T
	feature = feature.ravel()
	return feature

def preprocess_dataset():
	X_train=[]
	for tr in range(40):
		X=[]
		for i in range(1,6):
			f="s0"+str(i)+".dat"
			fname ="data/"+f
			x = cPickle.load(open(fname, 'rb'), encoding="bytes")
			eeg_realtime=x[b'data'][tr]  
			eeg_raw=np.reshape(eeg_realtime,(40,8064))
			eeg_raw=eeg_raw[:32,:]
			feature_vector=get_feature(eeg_raw) 
			feature_vector=feature_vector.reshape((320,1))
			X.append(feature_vector)
		X=np.array(X)
		X=X.reshape(-1,1)
		X_train.append(X)
	X_train=np.array(X_train)
	X_train=X_train.reshape(X_train.shape[1],-1)
	return X_train

def preprocess_y():
	fname="data/s01.dat"
	Y=[]
	x = cPickle.load(open(fname, 'rb'), encoding="bytes")
	for tr in range(40):
		label=x[b'labels'][tr]
		if(int(label[0])>5.5 and int(label[1])>5.5):
			Y.append([[1],[0],[0],[0],[0]])
		elif(int(label[0])>5.5 and int(label[1])<4.5):
			Y.append([[0],[0],[0],[1],[0]])
		elif(int(label[0])<4.5 and int(label[1])>5.5):
			Y.append([[0],[1],[0],[0],[0]])
		elif(int(label[0])<4.5 and int(label[1])<4.5):
			Y.append([[0],[0],[1],[0],[0]])
		else:
			Y.append([[0],[0],[0],[0],[1]])
	Y_train=np.array(Y)
	Y_train=Y_train.reshape(40,-1).T
	return Y_train	


def initialize(X,l1,l2,l3,nx):

    W1=np.random.randn(l1,nx)*0.01
    b1=np.zeros((l1,1))
    W2=np.random.randn(l2,l1)*0.01
    b2=np.zeros((l2,1))
    W3=np.random.randn(l3,l2)*0.01
    b3=np.zeros((l3,1))
    return W1,b1,W2,b2,W3,b3

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def forward(X,W1,b1,W2,b2,W3,b3):
    m=X.shape[1]
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=np.tanh(Z2)
    Z3=np.dot(W3,A2)+b3
    A3=sigmoid(Z3)
    not_parameters={"Z1":Z1,
                    "A1":A1,
                    "Z2":Z2,
                    "A2":A2,
                    "Z3":Z3,
                    "A3":A3
                    }
    return not_parameters

def backward(not_parameters,X,Y,W2,W3):
    m=X.shape[1]
    A3=not_parameters["A3"]
    Z3=not_parameters["Z3"]
    A2=not_parameters["A2"]
    Z2=not_parameters["Z2"]
    A1=not_parameters["A1"]
    Z1=not_parameters["Z1"]
    dZ3=A3-Y                    #1*m #1*m
    dW3=1/m*np.dot(dZ3,A2.T)     #1*m # m*4
    db3=1/m*np.sum(dZ3,axis=1,keepdims=True)       #1,m to #1,1
    dZ2=np.dot(W3.T,dZ3)*(1-np.power(np.tanh(Z2),2))  #4,1 #1,m
    dW2=1/m*np.dot(dZ2,A1.T)   #4,m #m,nx 
    db2=1/m*np.sum(dZ2,axis=1,keepdims=True)   #4,m to #4,1
    dZ1=np.dot(W2.T,dZ2)*(1-np.power(np.tanh(Z1),2))  #4,1 #1,m
    dW1=1/m*np.dot(dZ1,X.T)   #4,m #m,nx 
    db1=1/m*np.sum(dZ1,axis=1,keepdims=True)   #4,m to #4,1

    grads={"dW2":dW2,
           "dW3":dW3,
           "db3":db3,
           "db2":db2,
           "dW1":dW1,
           "db1":db1
           }
    return grads

def cost(A2,Y,X):
    m=X.shape[1]
    Y=Y.reshape(1,-1)
    A2=A2.reshape(1,-1)
    c=-1/m*np.sum(np.dot(Y,np.log(A2).T)+np.dot(1-Y,np.log(1-A2).T))
    return c

def iterate(X,Y,W1,b1,W2,b2,W3,b3,num,not_parameters,lr=0.2):
    for i in range(num):
        grads=backward(not_parameters,X,Y,W2,W3)
        dW1=grads["dW1"]
        db1=grads["db1"]
        dW2=grads["dW2"]
        db2=grads["db2"]
        dW3=grads["dW3"]
        db3=grads["db3"]
        W3=W3-lr*dW3
        b3=b3-lr*db3
        W2=W2-lr*dW2
        b2=b2-lr*db2
        W1=W1-lr*dW1
        b1=b1-lr*db1
        not_parameters=forward(X,W1,b1,W2,b2,W3,b3)
        if(i!=0 and i%2000==0):
            A3=not_parameters["A3"]
            print("Cost after %i iteration=%f"%(i, cost(A3,Y,X)))
        
    updated={"W2":W2,
             "b2":b2,
             "W1":W1,
             "b1":b1,
             "W3":W3,
             "b3":b3
             }
    return updated


def check_model(X,updated_parameters):
    w1=updated_parameters["W1"]
    w2=updated_parameters["W2"]
    B1=updated_parameters["b1"]
    B2=updated_parameters["b2"]
    w3=updated_parameters["W3"]
    B3=updated_parameters["b3"]
    not_parameters=forward(X,w1,B1,w2,B2,w3,B3)
    A3=not_parameters["A3"]
    Y_predict=np.where(A3<0.5,0,1)
    return Y_predict

if __name__=="__main__":
    X=preprocess_dataset()
    print(X.shape)
    Y=preprocess_y()
    print(Y.shape)
    W1,b1,W2,b2,W3,b3=initialize(X,8,8,5,X.shape[0])
    not_parameters=forward(X,W1,b1,W2,b2,W3,b3)
    updated_parameters=iterate(X,Y,W1,b1,W2,b2,W3,b3,50000,not_parameters)
    w1=updated_parameters["W1"]
    w2=updated_parameters["W2"]
    B1=updated_parameters["b1"]
    B2=updated_parameters["b2"]
    w3=updated_parameters["W3"]
    B3=updated_parameters["b3"]
    y_predict_train=check_model(X,updated_parameters)
    print("Training Accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - Y)) * 100))

#Z1=n_h,40
#A1=n_h,40
#A2=5,40
#dZ2=5,40-5,40
#dW2=5,40.40,n_h=5,n_h
#db2=5,40=5,1
#dZ1=n_h,5.5,40=n_h,40-n_h,40
#dW1=n_h,40.40,1600=n_h,1600
#db1=n_h,40
#cost=1,200.200,1+1,200.200,1









