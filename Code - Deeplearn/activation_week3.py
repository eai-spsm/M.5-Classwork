import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("M.5-Classwork/Code - Deeplearn/train.csv")

y = dataset['label']
x = dataset.drop('label', axis=1) / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

X = x_train.values.T
Y = y_train.values
X_val = x_val.values.T
Y_val = y_val.values

# ===== ACTIVATION FUNCTIONS (hidden layer, swap to compare) =====
def relu(z):
    return np.maximum(0, z)

def deriv_relu(z):
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def deriv_tanh(z):
    return 1 - np.tanh(z) ** 2

ACTIVATIONS = {
    'relu': (relu, deriv_relu),
    'sigmoid': (sigmoid, deriv_sigmoid),
    'tanh': (tanh, deriv_tanh),
}

def softmax(z):
    z = z - np.max(z, axis=0)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def one_hot(Y):
    m = len(Y)
    oh = np.zeros((10, m))
    oh[Y, np.arange(m)] = 1
    return oh

def get_predict(A2):
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
    def __init__(self, activation_name):
        self.W1, self.b1, self.W2, self.b2 = init_params()
        self.act, self.deriv_act = ACTIVATIONS[activation_name]

    def forward_prop(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.act(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def back_prop(self, Z1, A1, Z2, A2, X, Y, alpha):
        m = len(Y)
        dZ2 = A2 - one_hot(Y)
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2)
        dZ1 = np.dot(self.W2.T, dZ2) * self.deriv_act(Z1)
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1)

        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1

    # stops early once val accuracy hits target, timing each epoch so
    # activations can be compared on both epochs-needed and speed/epoch.
    def fit(self, X, Y, X_val, Y_val, alpha, n_epochs, batch_size=128, target_acc=0.85):
        m = X.shape[1]
        epoch_times = []
        epochs_to_target = None
        history = []   # (epoch, val_acc, cumulative_time) every 10 epochs
        total_time = 0.0
        for epoch in range(1, n_epochs+1):
            t0 = time.perf_counter()
            order = np.random.permutation(m)
            for start in range(0, m, batch_size):
                idx = order[start:start+batch_size]
                X_batch, Y_batch = X[:, idx], Y[idx]
                Z1, A1, Z2, A2 = self.forward_prop(X_batch)
                self.back_prop(Z1, A1, Z2, A2, X_batch, Y_batch, alpha)
            dt = time.perf_counter() - t0
            epoch_times.append(dt)
            total_time += dt

            _, _, _, A2_val = self.forward_prop(X_val)
            val_acc = get_accuracy(A2_val, Y_val)
            if epochs_to_target is None and val_acc >= target_acc:
                epochs_to_target = epoch
            if epoch % 10 == 0 or epoch == n_epochs:
                history.append((epoch, val_acc, total_time))
                print(f'  epoch {epoch} val_acc {val_acc:.4f} total_time {total_time:.3f}s')
        return {
            'epochs_to_target': epochs_to_target,
            'final_val_acc': val_acc,
            'avg_s_per_epoch': np.mean(epoch_times),
            'history': history,
        }

LEARNING_RATE = 0.1
BATCH_SIZE = 128
N_EPOCHS = 200
TARGET_ACC = 0.85

results = {}
for name in ACTIVATIONS:
    print(f'--- {name} ---')
    model = NeuralNet(name)
    results[name] = model.fit(X, Y, X_val, Y_val, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, TARGET_ACC)

print('\n===== COMPARISON =====')
print(f"{'activation':<10} {'epochs_to_85%':<15} {'final_val_acc':<15} {'avg_s/epoch':<12}")
for name, r in results.items():
    epochs = r['epochs_to_target'] if r['epochs_to_target'] is not None else f'>{N_EPOCHS}'
    print(f"{name:<10} {str(epochs):<15} {r['final_val_acc']:<15.4f} {r['avg_s_per_epoch']:<12.4f}")

print('\n===== HISTORY (epoch, val_acc, cumulative_time) every 10 epochs =====')
for name, r in results.items():
    print(f'{name}:')
    for epoch, acc, t in r['history']:
        print(f'  epoch {epoch:<4} acc {acc:.4f}  time {t:.3f}s')
