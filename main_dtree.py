import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Define entropy function
def entropy(y):
    # Get unique labels and their counts
    unique_labels, counts = np.unique(y, return_counts=True)

    # Calculate probabilities of labels
    probabilities = counts / counts.sum()

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


# Define information gain function
def information_gain(X_column, y, threshold):
    # Calculate parent entropy
    parent_entropy = entropy(y)

    # Create masks for left and right subsets
    left_mask = X_column < threshold
    right_mask = X_column >= threshold

    # Calculate the number of instances in left and right subsets
    n = len(y)
    n_l, n_r = left_mask.sum(), right_mask.sum()

    # Calculate entropy for left and right subsets
    e_l, e_r = entropy(y[left_mask]), entropy(y[right_mask])

    # Calculate child entropy
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

    # Calculate information gain
    ig = parent_entropy - child_entropy

    return ig


# Define best split function
def best_split(X, y):
    # Initialize the best information gain, best threshold, and best column
    best_ig = -np.inf
    best_threshold = None
    best_column = None

    # Loop over all columns
    for column in range(X.shape[1]):
        # Get unique values in the column
        thresholds = np.unique(X[:, column])

        # Loop over all unique values as potential thresholds
        for threshold in thresholds:
            # Calculate information gain
            ig = information_gain(X[:, column], y, threshold)

            # If this threshold gives a higher information gain than any we've seen before,
            # update best_ig, best_threshold, and best_column
            if ig > best_ig:
                best_ig = ig
                best_threshold = threshold
                best_column = column

    return best_column, best_threshold


# Define decision tree class
class DecisionTree:
    def __init__(self, max_depth=None):
        # Initialize with max_depth
        self.max_depth = max_depth

    def _best_split(self, X, y):
        # Call the global best_split function
        return best_split(X, y)

    def fit(self, X, y, depth=0):
        # If the tree is at maximum depth or all labels are the same, make this node a leaf node
        if depth == self.max_depth or entropy(y) == 0:
            self.is_leaf = True
            # Use the most common label as the label for this leaf node
            self.class_label = max(set(y), key=list(y).count)
            return

        self.is_leaf = False
        # Find the best split
        self.column, self.threshold = self._best_split(X, y)

        # Create masks for left and right subsets
        left_mask = X[:, self.column] < self.threshold
        right_mask = X[:, self.column] >= self.threshold

        # Create and fit left and right subtrees
        self.left = DecisionTree(self.max_depth)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right = DecisionTree(self.max_depth)
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

    def predict(self, X):
        # If this is a leaf node, return the stored class label
        if self.is_leaf:
            return self.class_label

        # If the feature at the column of this node is less than the threshold,
        # follow the left subtree, else follow the right subtree
        if X[self.column] < self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)


# Define train-test split function
def train_test_split(X, y, test_size=0.2):
    size = len(X)
    # Get random indices for test set
    test_indices = random.sample(range(size), int(size * test_size))
    # Get the rest of the indices for training set
    train_indices = [i for i in range(size) if i not in test_indices]
    # Split data into training and test sets
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test


# Load data
df = pd.read_csv('votes.csv')

# Convert party labels to numerical values
df['party'] = df['party'].map({'republican': 0, 'democrat': 1})

# Split data into features and target
y = df['party'].values
X = df.drop('party', axis=1).values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize and fit decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Predict on test set
y_pred = [tree.predict(x) for x in X_test]


# Define accuracy calculation function
def accuracy(y_true, y_pred):
    # Calculate accuracy as percentage of correct predictions
    return 100 * sum(y_true == y_pred) / len(y_true)


# Define confusion matrix calculation function
def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1
    return matrix


# Define precision, recall, and F1-score calculation functions
def precision_recall_f1(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    precision = np.diag(matrix) / matrix.sum(axis=0)
    recall = np.diag(matrix) / matrix.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# Define learning curve plotting function
def plot_learning_curve(X_train, y_train, X_test, y_test):
    training_sizes = np.linspace(0.1, 1.0, 10)
    train_acc = []
    test_acc = []
    for size in training_sizes:
        subset_size = int(size * len(X_train))
        tree = DecisionTree(max_depth=3)
        tree.fit(X_train[:subset_size], y_train[:subset_size])
        y_train_pred = [tree.predict(x) for x in X_train[:subset_size]]
        y_test_pred = [tree.predict(x) for x in X_test]
        train_acc.append(accuracy(y_train[:subset_size], y_train_pred))
        test_acc.append(accuracy(y_test, y_test_pred))
    plt.plot(training_sizes, train_acc, label='Train')
    plt.plot(training_sizes, test_acc, label='Test')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy (%)')
    plt.title('Learning curve')
    plt.legend()
    plt.show()

# Fit decision tree with full training set
tree.fit(X_train, y_train)

# Predict on test set
y_pred = [tree.predict(x) for x in X_test]

# Print size of training and test sets
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples\n')

# Print total accuracy
print(f'Total accuracy: {accuracy(y_test, y_pred):.2f}%\n')

# Print confusion matrix
print(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}\n')

# Print precision, recall, and F1-score values
precision, recall, f1 = precision_recall_f1(y_test, y_pred)
print('Class-wise metrics:\n')
print(f'Precision (Class 0): {precision[0]:.2f}')
print(f'Recall (Class 0): {recall[0]:.2f}')
print(f'F1-score (Class 0): {f1[0]:.2f}\n')

print(f'Precision (Class 1): {precision[1]:.2f}')
print(f'Recall (Class 1): {recall[1]:.2f}')
print(f'F1-score (Class 1): {f1[1]:.2f}\n')

# Print Macro-average and Weighted-average for precision, recall and F1-score
n_samples = len(y_test)
n_class0 = np.count_nonzero(y_test==0)
n_class1 = np.count_nonzero(y_test==1)

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

weighted_precision = (n_class0 * precision[0] + n_class1 * precision[1]) / n_samples
weighted_recall = (n_class0 * recall[0] + n_class1 * recall[1]) / n_samples
weighted_f1 = (n_class0 * f1[0] + n_class1 * f1[1]) / n_samples

print(f'Macro-Average Precision: {macro_precision:.2f}')
print(f'Macro-Average Recall: {macro_recall:.2f}')
print(f'Macro-Average F1-score: {macro_f1:.2f}\n')

print(f'Weighted-Average Precision: {weighted_precision:.2f}')
print(f'Weighted-Average Recall: {weighted_recall:.2f}')
print(f'Weighted-Average F1-score: {weighted_f1:.2f}\n')

# Plot learning curve
plot_learning_curve(X_train, y_train, X_test, y_test)
