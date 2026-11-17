import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("M.5-Classwork/Code - Deeplearn/train.csv")
# print(dataset.shape)
# print(dataset.head())

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

    # ===== STEP 6: TRAINING LOOP (mini-batch) =====
    # instead of one gradient step over ALL m samples, chop the data into
    # batch_size chunk and take one gradient step per chunk. More, noisier
    # steps per epoch instead of one smooth-but-slow step on the whole set.
    def fit(self, X, Y, alpha, n_epochs, batch_size=128):
        m = X.shape[1]
        for epoch in range(1, n_epochs+1):
            order = np.random.permutation(m)   # reshuffle each epoch so batches differ
            for start in range(0, m, batch_size):
                idx = order[start:start+batch_size]
                X_batch, Y_batch = X[:, idx], Y[idx]
                Z1, A1, Z2, A2 = self.forward_prop(X_batch)
                self.back_prop(Z1, A1, Z2, A2, X_batch, Y_batch, alpha)
            if epoch % 10 == 0 or epoch == n_epochs:
                _, _, _, A2_full = self.forward_prop(X)   # check on whole set, not just last batch
                accuracy = get_accuracy(A2_full, Y)
                print(f'Epoch: {epoch} Accuracy: {accuracy}')

X = x_train.values.T   # (784, m) — features as rows, samples as columns
Y = y_train.values     # (m,)

LEARNING_RATE = 0.1   # step size for gradient descent, tune if accuracy stalls/explodes
BATCH_SIZE = 128       # samples per gradient step, smaller = noisier but more steps/epoch

model = NeuralNet()
model.fit(X, Y, alpha=LEARNING_RATE, n_epochs=100, batch_size=BATCH_SIZE)

# ===== VISUAL CHECK: sample grid, prediction vs true label =====
# green label = model got it right, red = wrong. Lets you eyeball WHICH digit
# the model confuses, not just the overall accuracy number.
_, _, _, A2 = model.forward_prop(X)
pred = get_predict(A2)
accuracy = get_accuracy(A2, Y)

rows, cols = 9, 10
fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
for i, ax in enumerate(axes.flat):
    img = X[:, i].reshape(28, 28)
    ok = pred[i] == Y[i]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'{Y[i]} [{pred[i]}]', color='green' if ok else 'red', fontsize=8)
    ax.axis('off')
fig.suptitle(f'accuracy = {accuracy*100:.0f} %')
plt.tight_layout()
plt.show()