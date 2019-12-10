import sys;
import csv;
import statistics as sts;
import random as rnd;
import numpy as np;
from sklearn import svm
from sklearn import linear_model;
import os;
import traceback;
import pickle;

EXIT_FAILURE = 0x1;
EXIT_SUCCESS = 0x0;

VALIDATION_SPLIT = 0.7;

convert_to_int = lambda row: [int(cell) for cell in row];

def importer(method):
    def wrap(afname, *args, **kwargs):

        data = [];
        try:
            print("Importing " + afname);
            fp = open(afname, 'r');
            data = method(fp);
        except IOError:
            traceback.print_stack();
            sys.exit(EXIT_FAILURE);
        return data;
    return wrap;

@importer
def import_labels(fp):
    labels = [];
    for row in fp:
        labels.append(convert_to_int(row.split())[0x0]);
    return labels;

def transform_np(avFeatures, avLabels):
    avLabels = np.array(avLabels);
    avFeatures = np.array(avFeatures);
    return  avFeatures, avLabels;

def shuffle(avFeatures, avLables):

    mapper  = list(zip(avLables, avFeatures));
    rnd.shuffle(mapper);
    training_limit = int(len(mapper) * VALIDATION_SPLIT);
    avLables, avFeatures = zip(*mapper);

    avLables = np.array(avLables);
    avFeatures = np.array(avFeatures);

    return avLables[:training_limit], avFeatures[:training_limit], avLables[training_limit:], avFeatures[training_limit:];

def chi_sqr(X, y, topf):
    rows = len(X);
    cols = len(X[0]);
    O = [];
    for j in range(0, cols):
        ctable = [[1, 1], [1, 1], [1, 1]];
        for i in range(0, rows):
            if y[i] == 0:
                if X[i][j] == 0:
                    ctable[0][0] += 1
                elif X[i][j] == 1:
                    ctable[1][0] += 1
                elif X[i][j] == 2:
                    ctable[2][0] += 1
            elif y[i] == 1:
                if X[i][j] == 0:
                    ctable[0][1] += 1
                elif X[i][j] == 1:
                    ctable[1][1] += 1
                elif X[i][j] == 2:
                    ctable[2][1] += 1
        ctot = [sum(c) for c in ctable];
        rtot = [sum(c) for c in zip(*ctable)];
        tot = sum(ctot);
        exp = [[(r * c) / tot for r in rtot] for c in ctot];
        sqr = [[((ctable[i][j] - exp[i][j]) ** 2) / exp[i][j] for j in range(0, len(exp[0]))] for i in  range(0, len(exp))];
        chi2 = sum([sum(x) for x in zip(*sqr)])
        O.append(chi2);
    ind = sorted(range(len(O)), key=O.__getitem__, reverse=True);  # Sorts features from best to worse, in respect to chi2 values
    indi = ind[:topf];  # Returns the top 15 features column-indicies
    return indi


def feature_extraction(X, cols):
	V = []
	columns = list(zip(*X))
	for j in cols:
		V.append(columns[j])
	V = list(zip(*V))
	return V

@importer
def import_data(fp):
    data = [];
    for row in fp:
        data.append(convert_to_int(row.split()));
    return data;

def train_models(trainingData, trainingLabels):

    models = {
        'linsvc': svm.LinearSVC(),
        'svc': svm.SVC(gamma=0.001),
        'logreg': linear_model.LogisticRegression()
    };

    print("Training LinearSVC");
    models['linsvc'].fit(trainingData, trainingLabels);
    print("Training SVM.SVC")
    models['svc'].fit(trainingData, trainingLabels);
    print("Training Logistic Regression")
    models['logreg'].fit(trainingData, trainingLabels);

    pickle.dump(models, open('trained_models', 'wb'));

    return models;

def validate_models(validationData, validationLabels, models):

    predictions = {
        'linsvc': None,
        'svc': None,
        'logreg': None
    };
    print("Validation on LinearSVC")
    predictions['linsvc'] = models['linsvc'].predict(validationData);
    print("Validation on SVM.SVC")
    predictions['svc'] = models['svc'].predict(validationData);
    print("Validation on Logistic Regression")
    predictions['logreg'] = models['logreg'].predict(validationData);

    voted_predictions = np.array([]);
    for i in range(0, len(validationData)):
        voted_predictions = np.append(voted_predictions, sts.mode([predictions['linsvc'][i], predictions['svc'][i], predictions['logreg'][i]], ))

    sc = 0;
    for i in range(0, len(validationLabels)):
        if (voted_predictions[i] == validationLabels[i]):
            sc += 1;
    accuracy = (sc / len(validationLabels)) * 100;

    print('\nThe accuracy of this model on validation is {}%'.format(accuracy));

def test_models(testData, models):

    predictions = {
        'linsvc': models['linsvc'].predict(testData),
        'svc': models['svc'].predict(testData),
        'logreg': models['logreg'].predict(testData)
    };

    voted_predictions = np.array([]);
    for i in range(0, len(testData)):
        voted_predictions = np.append(voted_predictions, sts.mode([predictions['linsvc'][i], predictions['svc'][i], predictions['logreg'][i]], ))

    return voted_predictions;

def generate_report(fileName=""):

    try:
        fp = open(fileName, "w");
    except IOError as io:
        traceback.print_stack();
        sys.exit(EXIT_FAILURE);
    return fp;

if __name__ == '__main__':
    data = import_data(sys.argv[0x1]);
    labels = import_labels(sys.argv[0x2]);
    test = np.array(import_data(sys.argv[0x3]));

    report = generate_report("Selected_Features.txt");
    report1= generate_report("Predicted_Labels.txt");

    print("Transforming..")
    data, labels = transform_np(data, labels);

    print("Calculating chi-square")
    train_idx = chi_sqr(data, labels, 15);

    report.write("Selected feature indexes");
    report.write(str(train_idx));

    print("Shuffling")
    trainLabels, trainData, valLabels, valData = shuffle(data, labels);

    print("Feature extraction")
    trainingFeatures = feature_extraction(trainData, train_idx);
    print("Feature extraction")
    validationFeatures = feature_extraction(valData, train_idx);

    print("Training")
    if(os.path.exists('trained_models')):
        models = pickle.load(open('trained_models', 'rb'));
    else:
        models = train_models(trainData, trainLabels);

    print("Validate")
    validate_models(valData, valLabels, models);

    print("Predict");
    predictions = test_models(test, models);

    report1.write("Predicted values:");
    for idx in range(0,len(predictions),1):
        report1.write("\n"+ str(int(predictions[idx]))+"\t"+str(idx));

    report.close();
    report1.close();

    print("Done");