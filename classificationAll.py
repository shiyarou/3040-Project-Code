import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, exp

# ==========================================
# 1. LOAD & ENCODE ORIGINAL CATEGORICAL DATA
# ==========================================
# Assumes german.data is in the same folder.
# This is the original file with A11, A12, etc.
data = pd.read_csv("german.data", header=None, delim_whitespace=True)

# Last column is the class label (1 = good, 2 = bad)
X_df = data.iloc[:, :-1].copy()
y = data.iloc[:, -1].values  # should already be 1 or 2

# Encode categorical columns into integers
# (numeric columns are left as-is)
for col in X_df.columns:
    if X_df[col].dtype == "object":
        X_df[col] = X_df[col].astype("category").cat.codes

# Convert to numpy arrays
X = X_df.values.astype(float)

# ==========================================
# 2. TRAIN / TEST SPLIT (2/3 train, 1/3 test)
# ==========================================
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(len(X) * 2 / 3)

train_idx = indices[:split]
test_idx = indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ==========================================
# 3. KNN (NO SKLEARN)
# ==========================================
def knn_predict_one(x, X_train, y_train, k):
    """Predict label for a single sample x using KNN."""
    dists = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
    k_idx = np.argsort(dists)[:k]
    labels, counts = np.unique(y_train[k_idx], return_counts=True)
    return labels[np.argmax(counts)]

def knn_predict(X_eval, X_train, y_train, k):
    """Predict labels for all samples in X_eval using KNN."""
    return np.array([knn_predict_one(x, X_train, y_train, k) for x in X_eval])

# ==========================================
# 4. NAIVE BAYES (GAUSSIAN, NO SKLEARN)
# ==========================================
class GaussianNBManual:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            Xc = X[y == c]
            self.means[c] = Xc.mean(axis=0)
            self.vars[c] = Xc.var(axis=0) + 1e-6  # avoid divide-by-zero
            self.priors[c] = len(Xc) / len(X)

    def gaussian_prob(self, class_, x):
        mean = self.means[class_]
        var = self.vars[class_]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * pi * var)
        return numerator / denominator

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                likelihood = np.prod(self.gaussian_prob(c, x))
                prior = self.priors[c]
                posteriors[c] = likelihood * prior
            preds.append(max(posteriors, key=posteriors.get))
        return np.array(preds)

# ==========================================
# 5. DECISION TREE (ID3-LIKE, NO SKLEARN)
# ==========================================
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))

def split_dataset(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

def best_split(X, y):
    best_gain = -1
    best_feature, best_threshold = None, None
    current_entropy = entropy(y)
    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature, t)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            p = len(y_left) / len(y)
            gain = current_entropy - (p * entropy(y_left) + (1 - p) * entropy(y_right))

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

class DecisionTreeManual:
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        # If all labels are the same, return that label (leaf)
        if len(np.unique(y)) == 1:
            return y[0]

        feature, threshold = best_split(X, y)

        # If no split improves information gain, return majority class
        if feature is None:
            return np.bincount(y).argmax()

        node = {
            "feature": feature,
            "threshold": threshold
        }

        (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature, threshold)
        node["left"] = self._build_tree(X_left, y_left)
        node["right"] = self._build_tree(X_right, y_right)

        return node

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node  # leaf

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ==========================================
# 6. METRICS (NO SKLEARN)
# ==========================================
def confusion_matrix_manual(y_true, y_pred):
    """
    Confusion matrix for classes 1 and 2:
    [[actual=1,pred=1,  actual=1,pred=2],
     [actual=2,pred=1,  actual=2,pred=2]]
    """
    matrix = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t) - 1, int(p) - 1] += 1
    return matrix

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, positive_class=2):
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred, positive_class=2):
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))
    return tp / (tp + fn + 1e-9)

def f_score(p, r):
    return 2 * (p * r) / (p + r + 1e-9)

# ==========================================
# 7. PART B FIRST: KNN K=1..15 ERROR CURVES
#    AND AUTO-SELECT BEST K (MIN TEST ERROR)
# ==========================================
print("=== KNN error vs K (1 to 15) and auto-select best K (original data) ===")

K_values = range(1, 16)
train_errors = []
test_errors = []

for k in K_values:
    # Predict on training data to get training error
    y_pred_train = knn_predict(X_train, X_train, y_train, k)
    train_error = np.mean(y_pred_train != y_train)
    train_errors.append(train_error)

    # Predict on test data to get test error
    y_pred_test = knn_predict(X_test, X_train, y_train, k)
    test_error = np.mean(y_pred_test != y_test)
    test_errors.append(test_error)

    print(f"K={k:2d} | Train Error={train_error:.4f} | Test Error={test_error:.4f}")

# Pick best K as the one with minimum test error (break ties by smallest K)
best_k_index = int(np.argmin(test_errors))
best_k = list(K_values)[best_k_index]
print(f"\n>>> Best K based on minimum test error: K = {best_k} "
      f"(Test Error = {test_errors[best_k_index]:.4f})")

# Plot training vs test error
plt.figure(figsize=(10, 6))
plt.plot(list(K_values), train_errors, marker='o', label='Training Error')
plt.plot(list(K_values), test_errors, marker='s', label='Testing Error')

plt.title("KNN: Training vs Testing Error Rate (German Credit, original data)")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.xticks(list(K_values))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 8. PART A: USE BEST K IN KNN + NB + DT
# ==========================================
print("\n=== Classifier comparison using best K for KNN (original data) ===")
print(f"(Using KNN with K = {best_k})")

# KNN with best_k
print(f"\nRunning KNN with k={best_k} ...")
knn_pred = knn_predict(X_test, X_train, y_train, best_k)

print("Running Naive Bayes ...")
nb = GaussianNBManual()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("Running Decision Tree ... (might take a bit)")
dt = DecisionTreeManual()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

models_preds = {
    f"KNN (k={best_k})": knn_pred,
    "Naive Bayes": nb_pred,
    "Decision Tree": dt_pred
}

results = {}

for name, preds in models_preds.items():
    print("\n==============================")
    print(f"MODEL: {name}")
    cm = confusion_matrix_manual(y_test, preds)
    print("Confusion Matrix (rows=actual, cols=predicted; 1=Good, 2=Bad):")
    print(cm)

    acc = accuracy(y_test, preds)
    prec = precision(y_test, preds, positive_class=2)
    rec = recall(y_test, preds, positive_class=2)
    f = f_score(prec, rec)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (class=2/Bad): {prec:.4f}")
    print(f"Recall    (class=2/Bad): {rec:.4f}")
    print(f"F-score   (harmonic mean of P & R): {f:.4f}")

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f_score": f
    }

# Comparison table
print("\n#############################################")
print(" COMPARISON OF CLASSIFIERS (by F-score) ")
print("#############################################")
for name, stats in sorted(results.items(), key=lambda x: x[1]["f_score"], reverse=True):
    print(f"{name:18s} | Acc={stats['accuracy']:.4f}  "
          f"P={stats['precision']:.4f}  R={stats['recall']:.4f}  F={stats['f_score']:.4f}")
