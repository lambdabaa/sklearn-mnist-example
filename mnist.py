#!/usr/bin/env python3

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


class TextFormat:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def format(text, fmt):
    return fmt + str(text) + TextFormat.ENDC


def prepare_dataset():
    # 784 since these are 28x28 thumbnails.
    print('Download MNIST handwritten digits dataset...')
    X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Transform the 2d thumbnail arrays into a single dimension.
    X.reshape((X.shape[0], -1))
    # Scale features to fall between 0 and 1.
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    print('Partition data into train and test sets...')
    return train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=42)


def decision_tree(X_train, X_test, Y_train, Y_test, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, Y_train)
    print('Accuracy on train dataset: ' + format(model.score(X_train, Y_train), TextFormat.OKGREEN))
    print('Accuracy on test dataset: ' + format(model.score(X_test, Y_test), TextFormat.OKGREEN))
    return model


def naive_bayes(X_train, X_test, Y_train, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    print('Accuracy on train dataset: ' + format(model.score(X_train, Y_train), TextFormat.OKGREEN))
    print('Accuracy on test dataset: ' + format(model.score(X_test, Y_test), TextFormat.OKGREEN))
    return model


def mlp(X_train, X_test, Y_train, Y_test):
    model = MLPClassifier(hidden_layer_sizes=(98,))
    model.fit(X_train, Y_train)
    print('Accuracy on train dataset: ' + format(model.score(X_train, Y_train), TextFormat.OKGREEN))
    print('Accuracy on test dataset: ' + format(model.score(X_test, Y_test), TextFormat.OKGREEN))
    return model


def random_forest(X_train, X_test, Y_train, Y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)
    print('Accuracy on train dataset: ' + format(model.score(X_train, Y_train), TextFormat.OKGREEN))
    print('Accuracy on test dataset: ' + format(model.score(X_test, Y_test), TextFormat.OKGREEN))
    return model


def train_model(label, fn, X_train, X_test, Y_train, Y_test):
    print(format(label, TextFormat.UNDERLINE))
    fn(X_train, X_test, Y_train, Y_test)
    print()


def main():
    print(format('Dataset initialization', TextFormat.UNDERLINE))
    X_train, X_test, Y_train, Y_test = prepare_dataset()
    print('Will use ' +
          format(len(Y_train), TextFormat.BOLD) +
          ' for training and ' +
          format(len(Y_test), TextFormat.BOLD) +
          ' for testing...')
    print()
    model1 = train_model('Decision tree', decision_tree, X_train, X_test, Y_train, Y_test)
    model2 = train_model('Multilayer perceptron', mlp, X_train, X_test, Y_train, Y_test)
    model3 = train_model('Naive bayes', naive_bayes, X_train, X_test, Y_train, Y_test)
    model4 = train_model('Random forest', random_forest, X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
    main()
