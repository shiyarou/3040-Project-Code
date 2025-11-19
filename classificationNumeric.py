import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA (NUMERIC VERSION)
# ==========================================
# Assumes german.data-numeric is in the same folder.
data = pd.read_csv("german.data-numeric", header=None, delim_whitespace=True)
X = data.iloc[:, :-1].values      # features
y = data.iloc[:, -1].values       # labels (1 = good, 2 = bad)

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
# 4. CATEGORICAL NAIVE BAYES (NO SKLEARN)
#    Treats each feature as discrete (counts + Laplace smoothing)
# ==========================================
class CategoricalNBManual:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_counts_ = dict(zip(self.classes_, class_counts))
        self.class_log_prior_ = {
            c: np.log(count / len(y)) for c, count in self.class_counts_.items()
        }

        n_features = X.shape[1]
        self.n_features_ = n_features

        # Unique values per feature (over entire training set)
        self.feature_values_ = [np.unique(X[:, j]) for j in range(n_features)]
        self.n_values_ = [len(vals) for vals in self.feature_values_]

        # For each class and feature, store counts of each value
        self.feature_counts_ = {
            c: [dict() for _ in range(n_features)] for c in self.classes_
        }

        for c in self.classes_:
            X_c = X[y == c]
            for j in range(n_features):
                vals, counts = np.unique(X_c[:, j], return_counts=True)
                d = self.feature_counts_[c][j]
                for v, cnt in zip(vals, counts):
                    d[v] = cnt

        return self

    def _log_likelihood(self, x, c):
        log_prob = 0.0
        alpha = self.alpha
        class_count = self.class_counts_[c]

        for j in range(self.n_features_):
            v = x[j]
            counts_dict = self.feature_counts_[c][j]
            n_values = self.n_values_[j]
            count_v = counts_dict.get(v, 0)
            # Laplace-smoothed probability
            prob = (count_v + alpha) / (class_count + alpha * n_values)
            log_prob += np.log(prob)

        return log_prob

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            class_log_post = {}
            for c in self.classes_:
                log_prior = self.class_log_prior_[c]
                log_lik = self._log_likelihood(x, c)
                class_log_post[c] = log_prior + log_lik
            preds.append(max(class_log_post, key=class_log_post.get))
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
# 7. PART B: KNN K=1..15 ERROR CURVES + AUTO BEST K
# ==========================================
print("=== KNN error vs K (1 to 15) and auto-select best K (numeric data) ===")

K_values = range(1, 16)
train_errors = []
test_errors = []

for k in K_values:
    y_pred_train = knn_predict(X_train, X_train, y_train, k)
    train_error = np.mean(y_pred_train != y_train)
    train_errors.append(train_error)

    y_pred_test = knn_predict(X_test, X_train, y_train, k)
    test_error = np.mean(y_pred_test != y_test)
    test_errors.append(test_error)

    print(f"K={k:2d} | Train Error={train_error:.4f} | Test Error={test_error:.4f}")

best_k_index = int(np.argmin(test_errors))
best_k = list(K_values)[best_k_index]
print(f"\n>>> Best K based on minimum test error: K = {best_k} "
      f"(Test Error = {test_errors[best_k_index]:.4f})")

plt.figure(figsize=(10, 6))
plt.plot(list(K_values), train_errors, marker='o', label='Training Error')
plt.plot(list(K_values), test_errors, marker='s', label='Testing Error')

plt.title("KNN: Training vs Testing Error Rate (German Credit - numeric data)")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.xticks(list(K_values))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 8. PART A: USE BEST K IN KNN + CATEGORICAL NB + DT
# ==========================================
print("\n=== Classifier comparison using best K for KNN (numeric data) ===")
print(f"(Using KNN with K = {best_k})")

print(f"\nRunning KNN with k={best_k} ...")
knn_pred = knn_predict(X_test, X_train, y_train, best_k)

print("Running Categorical Naive Bayes ...")
nb = CategoricalNBManual(alpha=1.0)
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("Running Decision Tree ... (might take a bit)")
dt = DecisionTreeManual()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

models_preds = {
    f"KNN (k={best_k})": knn_pred,
    "Categorical NB": nb_pred,
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

print("\n#############################################")
print(" COMPARISON OF CLASSIFIERS (by F-score) ")
print("#############################################")
for name, stats in sorted(results.items(), key=lambda x: x[1]["f_score"], reverse=True):
    print(f"{name:18s} | Acc={stats['accuracy']:.4f}  "
          f"P={stats['precision']:.4f}  R={stats['recall']:.4f}  F={stats['f_score']:.4f}")
