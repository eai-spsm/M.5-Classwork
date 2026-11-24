import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Org - m5/Code - Deeplearn/train.csv")

y = dataset['label']
x = dataset.drop('label', axis=1) / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

X = x_train.values.T
Y = y_train.values
X_val = x_val.values.T
Y_val = y_val.values

# ===== ACTIVATION (sigmoid, all hidden layers) =====
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

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

class NeuralNet:
    # hidden_sizes: list of node counts, one per hidden layer (so len() = layer count)
    def __init__(self, hidden_sizes, n_in=784, n_out=10):
        sizes = [n_in] + list(hidden_sizes) + [n_out]
        self.W = [np.random.rand(sizes[i+1], sizes[i]) - 0.5 for i in range(len(sizes)-1)]
        self.b = [np.random.rand(sizes[i+1], 1) - 0.5 for i in range(len(sizes)-1)]

    def forward_prop(self, X):
        Z, A = [], [X]
        for i in range(len(self.W)):
            z = np.dot(self.W[i], A[-1]) + self.b[i]
            Z.append(z)
            is_output = i == len(self.W) - 1
            A.append(softmax(z) if is_output else sigmoid(z))
        return Z, A

    def back_prop(self, Z, A, Y, alpha):
        m = len(Y)
        n = len(self.W)
        dZ = A[-1] - one_hot(Y)   # output layer error
        for i in reversed(range(n)):
            dW = 1/m * np.dot(dZ, A[i].T)
            db = 1/m * np.sum(dZ)
            if i > 0:
                dZ = np.dot(self.W[i].T, dZ) * deriv_sigmoid(Z[i-1])
            self.W[i] = self.W[i] - alpha * dW
            self.b[i] = self.b[i] - alpha * db

    # full n_epochs run (no early stop), records (epoch, val_acc, cumulative_time) every 10 epochs
    def fit(self, X, Y, X_val, Y_val, alpha, n_epochs, batch_size=128, target_acc=0.85):
        m = X.shape[1]
        epoch_times = []
        epochs_to_target = None
        history = []
        total_time = 0.0
        for epoch in range(1, n_epochs+1):
            t0 = time.perf_counter()
            order = np.random.permutation(m)
            for start in range(0, m, batch_size):
                idx = order[start:start+batch_size]
                X_batch, Y_batch = X[:, idx], Y[idx]
                Z, A = self.forward_prop(X_batch)
                self.back_prop(Z, A, Y_batch, alpha)
            dt = time.perf_counter() - t0
            epoch_times.append(dt)
            total_time += dt

            _, A_val = self.forward_prop(X_val)
            val_acc = get_accuracy(A_val[-1], Y_val)
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

# per the slide: layers 1,2,3 x nodes-per-layer 10,100 (all 6 combos)
CONFIGS = {
    f'{L}x{N}': [N] * L
    for L in (1, 2, 3)
    for N in (10, 100)
}

results = {}
for name, hidden_sizes in CONFIGS.items():
    print(f'--- {name} ---')
    model = NeuralNet(hidden_sizes)
    results[name] = model.fit(X, Y, X_val, Y_val, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, TARGET_ACC)

print('\n===== COMPARISON =====')
print(f"{'layers':<10} {'epochs_to_85%':<15} {'final_val_acc':<15} {'avg_s/epoch':<12}")
for name, r in results.items():
    epochs = r['epochs_to_target'] if r['epochs_to_target'] is not None else f'>{N_EPOCHS}'
    print(f"{name:<10} {str(epochs):<15} {r['final_val_acc']:<15.4f} {r['avg_s_per_epoch']:<12.4f}")

print('\n===== HISTORY (epoch, val_acc, cumulative_time) every 10 epochs =====')
for name, r in results.items():
    print(f'{name}:')
    for epoch, acc, t in r['history']:
        print(f'  epoch {epoch:<4} acc {acc:.4f}  time {t:.3f}s')
