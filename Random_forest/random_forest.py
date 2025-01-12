import numpy as np
from one_tree import DecisionTreeClassifier
from collections import Counter
# Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# This methos is for creating the data instance for bagging
# Here the samples may repeate
# Bootstrap agregayiion 
# Then the each generated samole dataset is passed to one if the ensemble model
# in this case tree
# after the prediction is maid by each tree the resulting prediction will
# be cakcukated from averageing the all preds
def bootstrap_sampler(X,y): 
    n_samples =  X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

class RandomForestClassifier:

    def __init__(self, n_trees = 100, min_samples_split = 2, max_depth= 20, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth= max_depth
        self.n_feats = n_feats
        self.trees = []

        

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.trees = []
        for _ in tqdm(range(self.n_trees), desc="Training Random Forest"):
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                num_features=self.n_feats
            )
            X_sample, y_sample = bootstrap_sampler(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)



    def _plot_decision_boundary(self, X, y, ax):
        ax.clear()
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
        ax.set_title(f"Random Forest Decision Boundary (Trees: {len(self.trees)})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")


    def most_common_label(self, labels):
        label_counter = Counter(labels)
        most_common_label = label_counter.most_common(1)[0][0]
        return most_common_label
    

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0,1)
        y_pred = [self.most_common_label(tree_prediction) for tree_prediction in tree_predictions]
        return np.array(y_pred)
    






if __name__ == "__main__":

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
    random_forest = RandomForestClassifier(n_trees=10)
    random_forest.fit(features_train, labels_train)

    # Make predictions
    predictions = random_forest.predict(features_test)
    accuracy_score = calculate_accuracy(labels_test, predictions)

    print("Accuracy:", accuracy_score)
