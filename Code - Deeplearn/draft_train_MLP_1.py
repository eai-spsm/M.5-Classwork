import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("M.5-Classwork/Code - Deeplearn/train_clean.csv")
# print(dataset.shape)
# print(dataset.head())

# ===== STEP 1: DATA CLEAN (guard rail, not learning) =====
# Valid value in table [0:255] px
# Filter on data RELIABILITY, before any training even starts.
# bad cell = pixel out of [0,255] range or NaN.
pixel_cols = dataset.columns.drop('label')
bad = (dataset[pixel_cols] < 0) | (dataset[pixel_cols] > 255) | dataset[pixel_cols].isna()
bad_frac = bad.sum(axis=1) / len(pixel_cols)

dataset = dataset[bad_frac <= 0.05].copy()           # drop rows that are mostly anomalous
dataset[pixel_cols] = dataset[pixel_cols].clip(lower=0, upper=255)  # patch the rest

dataset.to_csv("M.5-Classwork/Code - Deeplearn/train_clean.csv", index=False)

y_train = dataset['label'] # Lable
x_train = dataset.drop('label', axis=1) # Featrue
x_train = x_train / 255.0  # normalize 0-255 -> 0-1, keep weight update stable

# image = x_train.iloc[8].values
# image = image.reshape(28,28)

# plt.imshow(image, cmap='gray')
# plt.title(f"Label: {y_train.iloc[8]}")
# plt.show()

# ===== STEP 2: ACTIVATION FUNCTIONS =====
# the "judge" at each neuron: look at the weighted score, decide pass or block.
# relu = hidden layer filter. Z>0 -> pass value through. Z<=0 -> blocked (output 0).
# also the reason net can learn curves, not just a straight line: stack of pure
# linear layers collapses into one linear layer, relu's kink breaks that.
def relu(z):
    return np.maximum(0, z)

def deriv_relu(z):   # slope of relu, used by back_prop to push error back through it
    return z > 0

def softmax(z): # stabel softmax
    # not a filter like relu, just squashes final score into a 0-1 probability
    # per class (sums to 1) so it's comparable to one_hot(Y) for the loss.
    z = z - np.max(z, axis=0)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def one_hot(Y):
    m = len(Y)
    oh = np.zeros((10, m))
    oh[Y, np.arange(m)] = 1
    return oh

def get_predict(A2):   # winner class = highest probability after softmax
    return np.argmax(A2, axis=0)

def get_accuracy(predict, Y):
    return np.sum(get_predict(predict) == Y) / len(Y)

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

class NeuralNet:
    def __init__(self, n_in=784, n_hidden=10, n_out=10):
        # W = per-input "volume knob", how much each pixel counts toward a neuron firing.
        # b = per-neuron trigger threshold shift, not tied to any single input.
        # both start random here, uniform in [-0.5, 0.5], reshaped by training below.
        self.W1, self.b1, self.W2, self.b2 = init_params()

    # ===== STEP 3: FORWARD PROP =====
    # push image through: weighted sum (W.X+b) then activation, layer by layer.
    # this is the "assessment" pass that produces a prediction (A2).
    def forward_prop(self, X):
        Z1 = np.dot(self.W1, X) + self.b1   # hidden neuron raw score
        A1 = relu(Z1)                       # filter: only positive score pass on | neuron activation
        Z2 = np.dot(self.W2, A1) + self.b2  # output neuron raw score | output layer
        A2 = softmax(Z2)                    # turn score into probability per digit | predicted
        return Z1, A1, Z2, A2

    # ===== STEP 4: BACK PROP =====
    # chain rule walk backward through the forward graph (A2<-Z2<-A1<-Z1<-X),
    # computing the slope (gradient) of the error wrt every W and b.
    # this does NOT update anything yet, just measures "which way is downhill".
    def back_prop(self, Z1, A1, Z2, A2, X, Y, alpha):
        m = len(Y)
        dZ2 = A2 - one_hot(Y)                       # error at output (predicted - true)
        dW2 = 1/m * np.dot(dZ2, A1.T)                # slope wrt W2
        db2 = 1/m * np.sum(dZ2)                      # slope wrt b2
        dZ1 = np.dot(self.W2.T, dZ2) * deriv_relu(Z1)  # push error back through relu
        dW1 = 1/m * np.dot(dZ1, X.T)                 # slope wrt W1
        db1 = 1/m * np.sum(dZ1)                      # slope wrt b1

        # ===== STEP 5: GRADIENT DESCENT =====
        # apply the slope: step every weight/bias a little opposite the gradient,
        # alpha = how big that step is. this is the actual "re-tune" / learning.
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1

    # ===== STEP 6: TRAINING LOOP =====
    # repeat forward -> backprop -> gradient descent n_iters time.
    # each loop = one full re-tune cycle, accuracy should climb over iterations.
    def fit(self, X, Y, alpha, n_iters):
        for i in range(1, n_iters+1):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            self.back_prop(Z1, A1, Z2, A2, X, Y, alpha)
            if i % 100 == 0 or i == n_iters:
                accuracy = get_accuracy(A2, Y)
                print(f'Iteration: {i} Accuracy: {accuracy}')

X = x_train.values.T   # (784, m) — features as rows, samples as columns
Y = y_train.values     # (m,)

LEARNING_RATE = 0.03   # step size for gradient descent, tune if accuracy stalls/explodes

model = NeuralNet()
model.fit(X, Y, alpha=LEARNING_RATE, n_iters=1000)