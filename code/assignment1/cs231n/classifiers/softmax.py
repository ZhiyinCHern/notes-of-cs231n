from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        s=np.dot(X[i],W)
        s-=np.max(s) # 数值处理，防止溢出
        p=np.exp(s)
        p/=np.sum(p) # 归一化

        loss-=np.log(p)[y[i]]

        dW[:,y[i]]-=X[i]
        dW+=X[i][:,None]*p

    loss=loss/num_train+reg*np.sum(W*W)
    dW=dW/num_train+2*reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    N=X.shape[0]
    
    S=X@W
    S-=np.max(S,axis=1)[:,None]     # 防止数据溢出
    P=np.exp(S)
    P/=np.sum(P,axis=1)[:,None]     # 归一化
    
    loss=-np.mean(np.log(P[np.arange(N),y]))
    loss+=reg*np.sum(W*W)
    
    P[np.arange(N),y]-=1            # 合并one-hot编码
    dW=(X.T)@P/N+2*reg*W

    return loss, dW