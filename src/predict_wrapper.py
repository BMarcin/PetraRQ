import argparse
import logging
import pickle
import sys
from itertools import repeat

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='Predict scores')
    parser.add_argument('stdin', nargs='?', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()
    stdin_data = args.stdin.read()

    # load saved sklearn model
    with open("./models/xgboost/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("./models/xgboost/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("./models/xgboost/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # delete empty lines
    stdin_data = stdin_data.strip()

    # transform input data
    input_data = vectorizer.transform(stdin_data.split('\n'))

    # predict
    predictions = model.predict_proba(input_data)

    # translate predictions to labels
    translated_probabilities = []
    for probabilities, labels in zip(predictions, repeat(labels)):
        score_lines = []
        for prob, label in zip(probabilities, labels):
            score_lines.append("{}:{:.9f}".format(label, prob))
        translated_probabilities.append(" ".join(score_lines))

    # print results
    print("\n".join(translated_probabilities))
