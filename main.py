import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

#  Use class 0 and class 1 for simplicity
X = X[y != 2]
y = y[y != 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Function to calculate entropy
def calculate_entropy(y):
    p = np.mean(y)  # Proportion of class 1
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# Function to calculate Information Gain (IG)
def information_gain(X, y, feature_idx):
    median_value = np.median(X[:, feature_idx])
    left_split = y[X[:, feature_idx] <= median_value]
    right_split = y[X[:, feature_idx] > median_value]

    parent_entropy = calculate_entropy(y)

    left_entropy = calculate_entropy(left_split)
    right_entropy = calculate_entropy(right_split)

    n = len(y)
    weighted_entropy = (len(left_split) / n) * left_entropy + (len(right_split) / n) * right_entropy

    return parent_entropy - weighted_entropy


# Function to calculate Fairness Gain (FG)
def fairness_gain(y, left_split, right_split):
    def discrimination(y):
        p = np.mean(y)
        return abs(0.5 - p)

    disc_parent = discrimination(y)
    disc_left = discrimination(left_split)
    disc_right = discrimination(right_split)

    n = len(y)
    fg = disc_parent - ((len(left_split) / n) * disc_left + (len(right_split) / n) * disc_right)
    return fg


# Function to calculate Fair Information Gain (FIG)
def fair_information_gain(X, y, feature_idx):
    median_value = np.median(X[:, feature_idx])
    left_split = y[X[:, feature_idx] <= median_value]
    right_split = y[X[:, feature_idx] > median_value]

    ig = information_gain(X, y, feature_idx)

    fg = fairness_gain(y, left_split, right_split)

    if fg == 0:
        return ig
    else:
        return ig * fg


# Train a traditional decision tree using Information Gain
clf_traditional = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_traditional.fit(X_train, y_train)

# Predict using the traditional decision tree
y_pred_traditional = clf_traditional.predict(X_test)
accuracy_traditional = accuracy_score(y_test, y_pred_traditional)

# Calculate Information Gain for traditional decision tree
ig_scores_traditional = []
for feature_idx in range(X_train.shape[1]):
    ig = information_gain(X_train, y_train, feature_idx)
    ig_scores_traditional.append(ig)
    print(f"Traditional Tree - Information Gain for Feature {feature_idx + 1}: {ig:.4f}")

print(f"\nAccuracy of Traditional Decision Tree: {accuracy_traditional:.4f}")


# Calculate Fair Information Gain for the modified decision tree
fig_scores = []
for feature_idx in range(X_train.shape[1]):
    fig = fair_information_gain(X_train, y_train, feature_idx)
    fig_scores.append(fig)
    print(f"Modified Tree - Fair Information Gain for Feature {feature_idx + 1}: {fig:.4f}")


clf_modified = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_modified.fit(X_train, y_train)

y_pred_modified = clf_modified.predict(X_test)
accuracy_modified = accuracy_score(y_test, y_pred_modified)

print(f"\nAccuracy of Modified Decision Tree: {accuracy_modified:.4f}")

# Comparing the IG and FIG scores
print("\nComparison of Traditional IG and Modified FIG for each feature:")
for i in range(len(ig_scores_traditional)):
    print(f"Feature {i + 1}: Traditional IG = {ig_scores_traditional[i]:.4f}, Modified FIG = {fig_scores[i]:.4f}")
