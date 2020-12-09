"""
This is an implementation of Recurrent Neural Network from scratch with numpy library.
For the construction, it uses hyperbolic tangent function for the activation function of the hidden layer.
And it uses an affine function composed with a softmax layer. The loss training the model is cross-entropy loss.
"""

import numpy as np

def to_one_hot(y, k):
    (n, t) = y.shape
    one_hot = np.zeros((n, t, k))
    for sample in range(n):
        one_hot[sample, np.arange(t), y[sample]] = 1
    return one_hot

def softmax(x):
    """
    A numerically stable version of the softmax function
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

class Module:
    def __init__(self):
        super().__init__()
        self.params = dict()
        self.grads = dict()
        self.children = dict()
        self.cache = dict()

    def _register_param(self, name: str, param: np.ndarray):
        """ the parameter can be accessed via self.params[name]
        the gradient can be accessed via self.grads[name]
        """
        assert isinstance(param, np.ndarray)
        self.params[name] = param
        self.grads[name] = np.zeros_like(param)

    def _register_child(self, name: str, child: 'Module'):
        """ the module can be acccessed via self.children[name]"""
        assert isinstance(child, Module)
        self.children[name] = child

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *g):
        raise NotImplementedError

    def named_parameters(self, base: tuple = ()):
        """recursively get all params in a generator"""
        assert self.params.keys() == self.grads.keys()
        for name in self.params:
            full_name = '.'.join(base + (name,))
            yield (full_name, self.params[name], self.grads[name])

        # recursively on others
        for child_name, child in self.children.items():
            yield from child.named_parameters(base=base + (child_name,))

def weight_init(fan_in, fan_out):

    a = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(low=-a, high=a, size=(fan_out, fan_in))

def clear_grad(model):
    """
    clear the gradient in the parameters and replace them with 0's
    """
    for name, param, grad in model.named_parameters():
        name_split = name.split(".")
        child_name = name_split[0]
        param_name = name_split[1]
        model.children[child_name].grads[param_name] = np.zeros_like(model.children[child_name].grads[param_name])

def update_param(model, lr):
    """
    update the parameters of the network
    """
    for name, param, grad in model.named_parameters():
        name_split = name.split(".")
        child_name = name_split[0]
        param_name = name_split[1]
        model.children[child_name].params[param_name] -= lr * model.children[child_name].grads[param_name]

data_file = "./language_model.txt"
data = np.loadtxt(data_file, dtype=np.int, delimiter=',')
X_all = to_one_hot(data, 4)
X = X_all[30:]
test_X = X_all[:30]

import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(Y, Y_hat):
    """
    @brief      Compute cross-entropy loss between labels and predictions
                averaged across the N instances
    @param      Y       ground-truth label of shape (N x T x K)
    @param      Y_hat   predictions of shape (N x T x K)
    @return     the average cross-entropy loss between Y and Y_hat
    """
    # IMPLEMENT ME
    N, T, K = Y.shape
    res = 0
    for i in range(N):
        for j in range(T):
            res -= Y[i][j].T @ np.log(Y_hat[i][j])
    return res / N

def generate_labels(X):
    """
    @brief      Takes in samples and generates labels.
    @param      X       Samples of sequence data (N x T x D)
    @return     Y, labels of shape (N x T x K)
    """
    # IMPLEMENT ME
    N, T, D = X.shape
    Y = np.zeros((N, T, D + 1)) # D + 1 for the label
    for i in range(N):
        Y[i, 0 : (T - 1), 0 : D] = X[i, 1 : T, :]
        last = np.zeros(D + 1)
        last[D] = 1
        Y[i][T - 1] = last
    return Y

class RNNCell(Module):
    def __init__(self, parameters):
        super().__init__()
        self._register_param('V', parameters['V'])
        self._register_param('W', parameters['W'])
        self._register_param('U', parameters['U'])
        self._register_param('c', parameters['c'])
        self._register_param('b', parameters['b'])

    def forward(self, x_t, h_prev):
        """
        @brief      Takes a batch of input at the timestep t with the previous hidden states and compute the RNNCell output
        @param      x_t    A numpy array as the input (N, in_features)
        @param      h_prev A numpy array as the previous hidden state (N, hidden_features)
        @return     The current hidden state (N, hidden_features) and the prediction at the timestep t (N, out_features)
        """
        # IMPLEMENT ME
        N, in_features = x_t.shape
        N, hidden_features = h_prev.shape
        Wh = h_prev @ self.params['W'].T
        Ux = x_t @ self.params['U'].T
        h_current = np.tanh(Wh + Ux + self.params['b'])
        Vh = h_current @ self.params['V'].T
        y_pred = Vh + self.params['c']
        for i in range(N):
            y_pred[i] = softmax(y_pred[i])
        return h_current, y_pred

    def backward(self, x_t, y_t, y_hat_t, dh_next, h_t, h_prev):
        """
        @brief      Compute and update the gradients for parameters of RNNCell at the timestep t
        @param      x_t      A numpy array as the input (N, in_features)
        @param      y_t      A numpy array as the target (N, out_features)
        @param      y_hat_t  A numpy array as the prediction (N, out_features)
        @param      dh_next  A numpy array as the gradient of the next hidden state (N, hidden_features)
        @param      h_t      A numpy array as the current hidden state (N, hidden_features)
        @param      h_prev   A numpy array as the previous hidden state (N, hidden_features)
        @return     The gradient of the current hidden state (N, hidden_features)
        """
        # IMPLEMENT ME
        N, H = h_t.shape
        self.grads['V'] += ((y_hat_t - y_t).T @ h_t)
        self.grads['c'] += np.sum((y_hat_t - y_t), axis = 0)
        dh_current = np.zeros((N, H))
        if (dh_next == np.zeros_like(dh_next)).all():
            for i in range(N):
                dh_current[i] = (1 - h_t[i] ** 2 )*np.dot(self.params['V'].T, y_hat_t[i] - y_t[i])
        else:
            for i in range(N):
                dh_current[i] = (1 - h_t[i] ** 2 ) * (np.dot(self.params['W'].T, dh_next[i]) + np.dot(self.params['V'].T, y_hat_t[i] - y_t[i]))
        self.grads['b'] += np.sum(dh_current, axis=0)
        self.grads['U'] += np.dot(dh_current.T, x_t)
        self.grads['W'] += np.dot(dh_current.T, h_prev)
        return dh_current

class RNN(Module):
    def __init__(self, d, h, k):
        """
        @brief      Initialize weight and bias
        @param      d   size of the input layer
        @param      h   size of the hidden layer
        @param      k   size of the output layer
        NOTE: Do not change this function or variable names; they are
            used for grading.
        """
        super().__init__()
        self.d = d
        self.h = h
        self.k = k

        parameters = {}
        wb = weight_init(d + h + 1, h)
        parameters['W'] = wb[:, :h]
        parameters['U'] = wb[:, h:h+d]
        parameters['b'] = wb[:, h+d]
        wb = weight_init(h + 1, k)
        parameters['V'] = wb[:, :h]
        parameters['c'] = wb[:, h]
        self._register_child('RNNCell', RNNCell(parameters))

    def forward(self, X):
        """
        @brief      Takes a batch of samples and computes the RNN output
        @param      X   A numpy array as the input of shape (N x T x D)
        @return     Hidden states (N x T x H), RNN's output (N x T x K)
        """
        # IMPLEMENT ME
        N, T, D = X.shape
        h_cur = np.zeros((N, T, self.h))
        y_pred = np.zeros((N, T, self.k))
        h_ini = np.zeros((N, self.h))
        for t in range(T):
            if t == 0:
                h_cur[:, t, :], y_pred[:, t, :] = self.children['RNNCell'].forward(X[:, t, :], h_ini)
            else:
                h_cur[:, t, :], y_pred[:, t, :] = self.children['RNNCell'].forward(X[:, t, :], h_cur[:, t-1, :])
        return (h_cur, y_pred)

    def backward(self, X, Y, Y_hat, H):
        """
        @brief      Backpropagation of the RNN
        @param      X      A numpy array as the input of shape (N x T x D)
        @param      Y      A numpy array as the ground truth labels of shape (N x T x K)
        @param      Y_hat  A numpy array as the prediction of shape (N x T x K)
        @param      H      A numpy array as the hidden states after the forward of shape (N x T x H)
        """
        # IMPLEMENT ME
        N, T, D = X.shape
        gradient = np.zeros((N, self.h))
        dU = np.zeros((self.h, self.d))
        dW = np.zeros((self.h, self.h))
        dV = np.zeros((self.k, self.h))
        db = np.zeros((self.h,))
        dc = np.zeros((self.k,))
        for t in reversed(range(T)):
            if t == 0:
                gradient= self.children['RNNCell'].backward(X[:, t, :], Y[:, t, :], Y_hat[:, t, :], gradient, \
                                                            H[:,t,:], np.zeros((N,self.h)))
            else:
                gradient = self.children['RNNCell'].backward(X[:, t, :], Y[:, t, :], Y_hat[:, t, :], gradient, H[:,t,:], H[:, t-1, :])
        return

def getsoftmax(output):
    y_hat = np.zeros(output.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            y_hat[i,j,:] = softmax(output[i,j,:])
    return y_hat

def train_one_epoch(model, X, test_X, lr):
    """
    @brief      Takes in a model and train it for one epoch.
    @param      model   The recurrent neural network
    @param      X       The features of training data (N x T x D)
    @param      test_X  The features of testing data (M x T x D)
    @param      lr      Learning rate
    @return     (train_cross_entropy, test_cross_entropy), the cross
                entropy loss for train and test data
    """
    # IMPLEMENT ME
    N, T, D = X.shape
    clear_grad(model)
    H, Y_hat = model.forward(X)
    Y = generate_labels(X)
    model.backward(X, Y, Y_hat, H)
    update_param(model, lr / N)
    H, output = model.forward(X)
    Y = generate_labels(X)
    Y_hat1 = getsoftmax(output)
    train_cross_entropy = cross_entropy(Y, Y_hat1)
    H, output = model.forward(test_X)
    Y = generate_labels(test_X)
    Y_hat2 = getsoftmax(output)
    test_cross_entropy = cross_entropy(Y, Y_hat2)
    return (train_cross_entropy, test_cross_entropy)

d = 4
h = 5
lr = 0.05

model = RNN(d, h, h)
epocs = 100
train_acc = np.zeros(epocs)
test_acc = np.zeros(epocs)
for i in range(epocs):
    train_acc[i], test_acc[i] = train_one_epoch(model, X, test_X, lr)
