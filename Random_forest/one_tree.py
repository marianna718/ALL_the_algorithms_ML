from collections import Counter
import numpy as np


def entropy(labels):
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root_node = None

    def fit(self, features, labels):
        self.num_features = features.shape[1] if not self.num_features else min(self.num_features, features.shape[1])
        self.root_node = self._grow_tree(features, labels)

    def predict(self, features):
        return np.array([self._traverse_tree(sample, self.root_node) for sample in features])

    def _grow_tree(self, features, labels, depth=0):
        num_samples, num_features = features.shape
        num_unique_labels = len(np.unique(labels))

        # Stopping criteria
        if depth >= self.max_depth or num_unique_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(labels)
            return TreeNode(value=leaf_value)

        feature_indices = np.random.choice(num_features, self.num_features, replace=False)

        # Find the best split
        best_feature_index, best_threshold = self._find_best_split(features, labels, feature_indices)

        # Split the data
        left_indices, right_indices = self._split(features[:, best_feature_index], best_threshold)
        left_subtree = self._grow_tree(features[left_indices, :], labels[left_indices], depth + 1)
        right_subtree = self._grow_tree(features[right_indices, :], labels[right_indices], depth + 1)

        return TreeNode(best_feature_index, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, features, labels, feature_indices):
        best_gain = -1
        best_feature_index, best_threshold = None, None

        for feature_index in feature_indices:
            feature_column = features[:, feature_index]
            unique_thresholds = np.unique(feature_column)
            for threshold in unique_thresholds:
                information_gain = self._calculate_information_gain(labels, feature_column, threshold)

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _calculate_information_gain(self, labels, feature_column, threshold):
        # Parent entropy
        parent_entropy = entropy(labels)

        # Split data
        left_indices, right_indices = self._split(feature_column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Weighted average entropy of the child nodes
        num_samples = len(labels)
        num_left, num_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = entropy(labels[left_indices]), entropy(labels[right_indices])
        weighted_child_entropy = (num_left / num_samples) * entropy_left + (num_right / num_samples) * entropy_right

        # Information gain
        info_gain = parent_entropy - weighted_child_entropy
        return info_gain

    def _split(self, feature_column, threshold):
        left_indices = np.argwhere(feature_column <= threshold).flatten()
        right_indices = np.argwhere(feature_column > threshold).flatten()
        return left_indices, right_indices

    def _traverse_tree(self, sample, node):
        if node.is_leaf_node():
            return node.value

        if sample[node.feature_index] <= node.threshold:
            return self._traverse_tree(sample, node.left)
        return self._traverse_tree(sample, node.right)

    def _most_common_label(self, labels):
        label_counter = Counter(labels)
        most_common_label = label_counter.most_common(1)[0][0]
        return most_common_label


if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def calculate_accuracy(true_labels, predicted_labels):
        accuracy_score = np.sum(true_labels == predicted_labels) / len(true_labels)
        return accuracy_score

    # Load dataset
    data = datasets.load_breast_cancer()
    feature_matrix, target_labels = data.data, data.target

    # Split dataset
    features_train, features_test, labels_train, labels_test = train_test_split(
        feature_matrix, target_labels, test_size=0.2, random_state=1234
    )

    # Train decision tree classifier
    decision_tree = DecisionTreeClassifier(max_depth=10)
    decision_tree.fit(features_train, labels_train)

    # Make predictions
    predictions = decision_tree.predict(features_test)
    accuracy_score = calculate_accuracy(labels_test, predictions)

    print("Accuracy:", accuracy_score)
