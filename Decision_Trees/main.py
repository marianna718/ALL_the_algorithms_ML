import numpy as np
from tqdm import tqdm
import time

class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None):
        self.criterion = criterion  # "gini" or "entropy"
        self.max_depth = max_depth  # Limit the depth of the tree
        self.tree = None  # Store the tree structure
        print("hellou init")


    def take_data(self, X, y):
        """
        Prepares the dataset for training.
        """
        # Assuming X and y are numpy arrays or can be converted to numpy
        X = np.array(X)
        y = np.array(y)
        return X, y

    def entropy_calc(self, y):
        """
        Calculates the entropy of a node.
        """
        # use this approach below for the non-integers
        # unique_labels, counts = np.unique(y, return_counts=True)

        class_counts = np.bincount(y)  # Count occurrences of each class
        # be aware, np.bincounts wont work with negatives and strings 
        # print(class_counts,"-hte class_count with bincount ")
        probabilities = class_counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])  # Avoid log(0)

    def gini(self, y):
        """
        Calculates the Gini index of a node.
        """
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return 1 - np.sum([p**2 for p in probabilities])

    def split_chooser(self, X, y):
        """
        Finds the best feature and threshold to split the data.
        """
        best_feature, best_threshold = None, None
        best_score = float("inf")  # Lower is better (minimize impurity)
        
        for feature_index in range(X.shape[1]):  # Loop through features
            thresholds = np.unique(X[:, feature_index])  # Unique values as thresholds
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices
                left_y, right_y = y[left_indices], y[right_indices]

                # Skip invalid splits
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # Calculate impurity of the split
                if self.criterion == "gini":
                    impurity = (len(left_y) / len(y)) * self.gini(left_y) + \
                               (len(right_y) / len(y)) * self.gini(right_y)
                elif self.criterion == "entropy":
                    impurity = (len(left_y) / len(y)) * self.entropy_calc(left_y) + \
                               (len(right_y) / len(y)) * self.entropy_calc(right_y)

                # Update best split if this one is better
                if impurity < best_score:
                    best_score = impurity
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    


    def fit(self, X, y):
        """
        Fits the decision tree to the data with animation.
        """
        print("hellou")
        X, y = self.take_data(X, y)
        print("Starting to build the decision tree...\n")
        time.sleep(1)  # Initial delay for effect

        # Simulate progress bar for the number of features
        for _ in tqdm(range(X.shape[1]), desc="Processing features"):
            time.sleep(0.5)  # Simulate processing time for each feature
        
        # Simulate recursive tree-building animation
        self.tree = self._build_tree(X, y, depth=0, animation=True)

        print("\nTree building complete!")

    def _build_tree(self, X, y, depth, animation=False):
        """
        Recursively builds the tree with optional animation.
        """
        if len(np.unique(y)) == 1:  # Pure node
            return {"type": "leaf", "class": y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return {"type": "leaf", "class": np.bincount(y).argmax()}

        feature, threshold = self.split_chooser(X, y)
        if feature is None:
            return {"type": "leaf", "class": np.bincount(y).argmax()}

        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices

        if animation:
            print(f"Depth: {depth} | Feature: {feature} | Threshold: {threshold}")
            time.sleep(0.5)  # Simulate tree building

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1, animation)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1, animation)

        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    # def fit(self, X, y):
    #     """
    #     Fits the decision tree to the data.
    #     """
    #     X, y = self.take_data(X, y)
    #     self.tree = self._build_tree(X, y, depth=0)

    # def _build_tree(self, X, y, depth):
    #     """
    #     Recursively builds the tree.
    #     """
    #     # Stopping conditions
    #     if len(np.unique(y)) == 1:  # Pure node
    #         return {"type": "leaf", "class": y[0]}
    #     if self.max_depth is not None and depth >= self.max_depth:
    #         return {"type": "leaf", "class": np.bincount(y).argmax()}  # Majority class

    #     # Find the best split
    #     feature, threshold = self.split_chooser(X, y)
    #     if feature is None:
    #         return {"type": "leaf", "class": np.bincount(y).argmax()}  # Majority class

    #     # Split the data
    #     left_indices = X[:, feature] <= threshold
    #     right_indices = ~left_indices
    #     left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
    #     right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

    #     return {
    #         "type": "node",
    #         "feature": feature,
    #         "threshold": threshold,
    #         "left": left_tree,
    #         "right": right_tree,
    #     }

    def predict_instance(self, instance, node):
        """
        Predicts the class of a single instance.
        """
        if node["type"] == "leaf":
            return node["class"]
        if instance[node["feature"]] <= node["threshold"]:
            return self.predict_instance(instance, node["left"])
        else:
            return self.predict_instance(instance, node["right"])

    def predict(self, X):
        """
        Predicts the class labels for multiple instances.
        """
        return [self.predict_instance(instance, self.tree) for instance in X]