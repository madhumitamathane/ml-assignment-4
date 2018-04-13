import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

DEFAULT_PATH   = "./ThoraricSurgery.csv"
DATA_HEADERS   = [
    "DGN", 
    "PRE4", 
    "PRE5", 
    "PRE6", 
    "PRE7", 
    "PRE8", 
    "PRE9", 
    "PRE10", 
    "PRE11",
    "PRE14", 
    "PRE17", 
    "PRE19", 
    "PRE25", 
    "PRE30", 
    "PRE32", 
    "AGE", 
    "Risk1Y"
]
DUMMY_COLUMNS  = [
    'DGN', 
    'PRE6',
    'PRE7',
    'PRE8',
    'PRE9', 
    'PRE10', 
    'PRE11', 
    'PRE14',
    'PRE17', 
    'PRE19', 
    'PRE25',
    'PRE30', 
    'PRE32'
]
Y_LABELS       = [
    "T",
    "F"
]

def read_data(path):
    df = pd.read_csv(path)
    df = prepare_data(df)
    return df

def prepare_data(df):
    df.columns = DATA_HEADERS
    df = pd.get_dummies(df, columns = DUMMY_COLUMNS)
    return df

def get_pretty_confusion_table(cmat, y_labels):
    lines = []
    space = "{0:<10}".format("")
    header = "".join(["{0:<10}".format(y) for y in y_labels])

    lines.append(space + header)

    for row_index, row in enumerate(cmat):
        row_str = "{0:<10}".format(y_labels[row_index])
        row_str += "".join(["{0:<10}".format(x) for x in row])
        lines.append(row_str)

    return "\n".join(lines)

def print_pretty_confusion_table(cmat, y_labels):
    print("Confustion Matrix (actuals x predicted)")
    print("")
    print(get_pretty_confusion_table(cmat, y_labels))

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    print("Reading data from '{0}'...".format(path))
    print("")

    df = read_data(path)

    X = df.loc[:, df.columns != "Risk1Y"]
    y = df.loc[:, "Risk1Y"]


    print("Performing test/train split for cross validation (random_state is fixed)...")
    print("")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    print("====== Results =========================")

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    y_pred = logreg.predict(X_test)
    cmat = confusion_matrix(y_test, y_pred, labels = Y_LABELS)
    print("")
    print_pretty_confusion_table(cmat, Y_LABELS)
    print("========================================")

if __name__ == "__main__":
    main()