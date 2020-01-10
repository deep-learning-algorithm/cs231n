# -*- coding: utf-8 -*-

# @Time    : 19-7-16 上午10:24
# @Author  : zj


import numpy as np


class LogisticClassifier(object):

    def __init__(self):
        self.W = None
        self.b = None

        self.lr = None
        self.reg = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        self.lr = learning_rate
        self.reg = reg

        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim)
            self.b = np.zeros(1)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, dW, db = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        scores = self.logistic_regression(X)
        y_pred = (scores > 0.5).astype(np.uint8)

        return y_pred

    def loss(self, X_batch, y_batch, reg, delta=1):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        eplison = 1e-5
        num_train = X_batch.shape[0]

        scores = self.logistic_regression(X_batch)
        data_loss = -1.0 / num_train * \
                    np.sum(y_batch * np.log(np.maximum(scores, eplison)) + (1 - y_batch) * np.log(
                        np.maximum(1 - scores, eplison)))
        reg_loss = 0.5 * reg * np.sum(self.W ** 2)

        loss = data_loss + reg_loss

        dscores = scores - y_batch
        dscores /= num_train
        dW = X_batch.T.dot(dscores) + reg * self.W
        db = np.sum(dscores)

        return loss, dW, db

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def logistic_regression(self, x):
        """
        :param x: A numpy array of shape (N, D)
        :param w: A numpy array of shape (D)
        :param b: A numpy array of shape (1)
        :return: A numpy array of shape (N)
        """
        z = x.dot(self.W) + self.b
        return self.sigmoid(z)
