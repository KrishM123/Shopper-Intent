import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    f = open(filename).read()
    f = f.split('\n')
    f = f[1:-1]
    labels = []
    for num in range(0, len(f)):
        f[num] = f[num].split(',')
        f[num][0] = int(f[num][0])
        f[num][1] = float(f[num][1])
        f[num][2] = int(f[num][2])
        f[num][3] = float(f[num][3])
        f[num][4] = int(f[num][4])
        f[num][5] = float(f[num][5])
        f[num][6] = float(f[num][6])
        f[num][7] = float(f[num][7])
        f[num][8] = float(f[num][8])
        f[num][9] = float(f[num][9])
        months = {'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'June':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}
        f[num][10] = months[f[num][10]]
        f[num][11] = int(f[num][11])
        f[num][12] = int(f[num][12])
        f[num][13] = int(f[num][13])
        f[num][14] = int(f[num][14])
        visitor = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 1}
        f[num][15] = visitor[f[num][15]]
        weekend = {'TRUE': 1, 'FALSE':0}
        f[num][16] = weekend[f[num][16]]
        labels.append(f[num][-1])
        f[num] = f[num][:-1]
    return (f, labels)


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    tru = 0
    fal = 0
    for ele in labels:
        if ele == 'TRUE':
            tru += 1
        if ele == 'FALSE':
            fal += 1
    tp = 0
    tn = 0
    for pos in range(0, len(predictions)):
        if labels[pos] == predictions[pos]:
            if labels[pos] == 'TRUE':
                tp += 1
            if labels[pos] == 'FALSE':
                tn += 1

    return tp/tru, tn/fal


if __name__ == "__main__":
    main()
