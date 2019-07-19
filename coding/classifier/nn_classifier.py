# -*- coding: utf-8 -*-

# @Time    : 19-7-18 上午10:15
# @Author  : zj


import numpy as np


class NN(object):

    def __init__(self, hidden_dims, input_dim=32 * 32 * 3, num_classes=10, weight_scale=1e-2, learning_rate=1e-3,
                 reg=0.0, dtype=np.float64):
        self.reg = reg
        self.lr = learning_rate
        self.dtype = dtype
        self.params = {}
        self.num_layers = len(hidden_dims) + 1

        if hidden_dims is None:
            self.params['W1'] = weight_scale * np.random.randn(input_dim, num_classes)
            self.params['b1'] = np.zeros((1, num_classes))
        else:
            for i in range(self.num_layers):
                if i == 0:
                    in_dim = input_dim
                    out_dim = hidden_dims[i]
                elif i == (self.num_layers - 1):
                    in_dim = hidden_dims[i - 1]
                    out_dim = num_classes
                else:
                    in_dim = hidden_dims[i - 1]
                    out_dim = hidden_dims[i]

                self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(in_dim, out_dim)
                self.params['b%d' % (i + 1)] = np.zeros((1, out_dim))

        self.configs = {}
        config = {'learning_rate': learning_rate}
        for k in self.params.keys():
            self.configs[k] = config.copy()

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def train(self, X, y, num_iters=100, batch_size=200, verbose=False):
        """
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        X = X.astype(self.dtype)
        num_train, dim = X.shape

        # Run stochastic gradient descent to optimize W
        loss_history = []
        range_list = np.arange(0, num_train, step=batch_size)
        for it in range(num_iters):
            total_loss = 0
            for i in range_list:
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # evaluate loss and gradient
                loss, grads = self.loss(X_batch, y_batch)
                total_loss += loss

                for k in self.params.keys():
                    #     config = self.configs[k]
                    w = self.params[k]
                    dw = grads[k]

                    # next_w, next_config = self.adam(w, dw, config)
                    next_w = w - self.lr * dw

                    self.params[k] = next_w
                    # self.configs[k] = config

            avg_loss = total_loss / len(range_list)
            loss_history.append(avg_loss)

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, avg_loss))

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
        scores, caches = self.forward(X)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        y_pred = np.argmax(probs, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        scores, caches = self.forward(X_batch)
        data_loss, dout = self.softmax_loss(scores, y_batch)

        reg_loss = 0
        for i in range(self.num_layers):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W%d' % (i + 1)] ** 2)
        loss = data_loss + reg_loss

        grads = {}
        dx = None
        for i in reversed(range(self.num_layers)):
            cache = caches['cache%d' % (i + 1)]
            if i == (self.num_layers - 1):
                dx, dw, db = self.affine_backward(dout, cache)
            else:
                dx, dw, db = self.affine_relu_backward(dx, cache)
            grads['W%d' % (i + 1)] = dw + self.reg * self.params['W%d' % (i + 1)]
            grads['b%d' % (i + 1)] = db

        return loss, grads

    def forward(self, X):
        a = None
        z = None
        caches = {}
        for i in range(self.num_layers):
            if i == 0:
                a = X
            if i == (self.num_layers - 1):
                z, caches['cache%d' % self.num_layers] = self.affine_forward(a,
                                                                             self.params['W%d' % (self.num_layers)],
                                                                             self.params['b%d' % (self.num_layers)])
            else:
                a, caches['cache%d' % (i + 1)] = self.affine_relu_forward(a,
                                                                          self.params['W%d' % (i + 1)],
                                                                          self.params['b%d' % (i + 1)])

        scores = z
        return scores, caches

    def affine_relu_forward(self, x, w, b):
        """
        Convenience layer that perorms an affine transform followed by a ReLU

        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer

        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = self.affine_forward(x, w, b)
        out, relu_cache = self.relu_forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    def affine_relu_backward(self, dout, cache):
        """
        Backward pass for the affine-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = self.relu_backward(dout, relu_cache)
        dx, dw, db = self.affine_backward(da, fc_cache)
        return dx, dw, db

    def affine_forward(self, x, w, b):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        inputs = x.reshape(x.shape[0], -1)
        out = inputs.dot(w) + b.reshape(1, -1)

        cache = (x, w, b)
        return out, cache

    def affine_backward(self, dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache

        dx = dout.dot(w.T).reshape(x.shape)
        dw = x.reshape(x.shape[0], -1).T.dot(dout)
        db = np.sum(dout, axis=0)

        return dx, dw, db

    def relu_forward(self, x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = x.copy()
        out[x < 0] = 0

        cache = x
        return out, cache

    def relu_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache

        dx = dout
        dx[x < 0] = 0

        return dx

    def softmax_loss(self, scores, y):
        num = y.shape[0]

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        data_loss = -1.0 / num * np.sum(np.log(probs[range(num), y]))

        dscores = scores
        dscores[range(num), y] -= 1
        dscores /= num

        return data_loss, dscores

    def adam(self, w, dw, config=None):
        """
        Uses the Adam update rule, which incorporates moving averages of both the
        gradient and its square and a bias correction term.

        config format:
        - learning_rate: Scalar learning rate.
        - beta1: Decay rate for moving average of first moment of gradient.
        - beta2: Decay rate for moving average of second moment of gradient.
        - epsilon: Small scalar used for smoothing to avoid dividing by zero.
        - m: Moving average of gradient.
        - v: Moving average of squared gradient.
        - t: Iteration number.
        """
        if config is None: config = {}
        config.setdefault('learning_rate', 1e-3)
        config.setdefault('beta1', 0.9)
        config.setdefault('beta2', 0.999)
        config.setdefault('epsilon', 1e-8)
        config.setdefault('m', np.zeros_like(w))
        config.setdefault('v', np.zeros_like(w))
        config.setdefault('t', 0)

        t = config['t'] + 1
        m = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
        mt = m / (1 - config['beta1'] ** t)
        v = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw ** 2)
        vt = v / (1 - config['beta2'] ** t)

        next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

        config['t'] = t
        config['m'] = m
        config['v'] = v

        return next_w, config
