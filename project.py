# Group: Iskandar Askarov, Ketki Ambekar
# UCID: ia253, kpa9
# Email: ia253@njit.edu, kpa9@njit.edu
# Written in Python 3.6
# Expected Runtime of project:
# Arguments for this program:
# 1) Training Data.gza
# 2) Test Data.gz
# 3) Training labels.txt

import sys
import traceback
import timeit
import numpy as np
from sklearn import svm;
from sklearn import model_selection

# #Read Args
# file1_train = sys.argv[0]
# file2_test = sys.argv[1]
# file3_trainLabels = sys.argv[2]

# Read Args
file1_train = sys.argv[1]
file2_test = sys.argv[2]
file3_trainLabels = sys.argv[3]

# Global Variables
startTime = timeit.default_timer()
EXIT_FAILURE = 0
EXIT_SUCCESS = 1
TrainData = None
TestData = None
TrainLabels = {}
TrainLabelsList = []
Val_Data = []
TrainData1 = None


# Step 1: Read Data. Train -> List , Test -> List, Labels (=target) -> dict
def readList(fileName=""):
    returnList = []
    i = 0
    rowVec = []
    nom = 0
    tempFile = None
    try:
        tempFile = (open(fileName, 'r')).read()  # open file in read only mode
        # Recording number of rows
        # No_of_rows=len(tempFile.splitlines())

        for row in tempFile.splitlines():
            rowVec = []
            # rowVec= [float(item) for item in row.split()]
            for item in row.split():
                rowVec.append(float(item))

            # Recording highest no. of columns for ther purpose of knowing how many random weights to choose later.
            # if len(rowVec) > No_of_columns:
            # No_of_columns   = len(rowVec)
            returnList.append(rowVec)
    except:
        print("Unable to open {} File".format(fileName));
        # traceback.print_exc();
        print("Exiting");
        sys.exit(EXIT_FAILURE);
    finally:
        a = 1
        # if(tempFile):
        #     tempFile.close();
    return returnList


def readList_np(fileName=""):
    returnList = []
    try:
        returnList = np.loadtxt(fileName, float)
    except:
        print("Unable to open {} data".format(fileName));
        traceback.print_exc();
        print("Exiting");
        sys.exit(EXIT_FAILURE);
    finally:
        a = 1
    return returnList


def readDict(fileName=""):
    tempFile = None;
    returnDict = dict();

    try:
        # tempFile = (open(fileName, 'r')).read()
        tempFile = (open(fileName))
        for line in tempFile:
            row = line.split();
            returnDict[int(row[0x1])] = int(row[0x0]);
    except FileNotFoundError as ex:
        print("Unable to open {} data".format(fileName));
        traceback.print_exc();
        print("Exiting");
        sys.exit(EXIT_FAILURE);

    finally:
        if (tempFile):
            tempFile.close();

    return returnDict;


def convertDict2List(my_dict={}):
    returnList = []

    for k, v in my_dict.items():
        returnList.append([k, v])

    return returnList


# Step 2: Split Train Data into 2 Subsets (row wise) -> "train" (70%) and "Validation" (30%)
def dataSpilt(part1_percent=0.0, trainDataSet=np.array):
    end = len(trainDataSet)
    mid = end * (part1_percent / 100)
    return_trainData = np.array()
    return_valData = np.array()

    try:
        for idx in (0, mid, 1):
            return1 = np.append(return1, trainDataSet[idx])
    except:
        a = 1
    finally:
        a = 1
    return return_trainData, return_valData


# Step 3: Calc Correlation of each column w.r.t the "target" ie the labels and arrange in Descending order of absolute magnitude.
def getPearsonCorrelation(List1, List2):
    corr = 0.0
    corr = (getSpread(List1) * getSpread(List2)) / (getSD(List1) * getSD(List2))
    corr = abs(corr)
    return corr


def getSpread(List1=[]):
    mean = sum(List1) / len(List1)
    returnVar = 0.0
    for k in List1:
        returnVar += (List1[k] - mean)
    return (returnVar)


def getSD(List1=[]):
    mean = sum(List1) / len(List1)
    returnVar = 0.0
    for k in List1:
        returnVar += (List1[k] - mean) ** 2

    return (returnVar ** 0.5)


def calcCorrelations(dataSet=np.array, targetLabels=np.array):
    scores = []

    for j in np.transpose(dataSet):
        scores.append(getPearsonCorrelation(dataSet[:, j]), targetLabels[0:, ])

    scores.sort(reverse=True)
    return scores


# Step 4: Extract the top x Columns
def getTopFeatures(No_of_features=0, feature_ranks=[]):
    return feature_ranks[0:No_of_features]


# Step 6: Train your SVM on extracted features
def trainSVM(finalFeatures=np.array, target=np.array, val_features=np.array, val_target=np.array):
    val_score = None
    clf = svm.LinearSVC()
    clf.fit(finalFeatures, target)

    # validation
    val_score = model_selection(clf, val_features, val_target, cv=5)
    print('Error for the Validation Data is {}\nMean Error for the Validation Data is {}'.format(val_score,
                                                                                                 val_score.mean()));
    print("Accuracy: %0.2f (+/- %0.2f)" % (val_score.mean(), val_score.std() * 2))

    return clf


# Step 7: Predict for test data
def predict(testData=np.array, clf=svm.LinearSVC):
    print("The predicted labels for test data are:")
    for k in clf.predict(testData):
        print(str(k))


# Main Function
if __name__ == '__main__':
    Features_ranking = None
    Features_selected = None

    # Read Files
    TrainLabels = readDict(file3_trainLabels)
    TrainData = readList_np(file1_train)
    TestData = readList(file2_test)

    # Convert Test labels from dictonary to numpy array
    TrainLabelsList = np.array(convertDict2List(TrainLabels), float)

    # Calculate pearson Correlation of each column with target label data
    Feature_ranking = calcCorrelations(TrainData, TrainLabelsList)

    # Get top 20 Features
    Features_selected = getTopFeatures(20, Feature_ranking)
    print("Features Selected are: " + str(Features_selected))

    # Split  training dataset into Train data and Validation Data
    tempObject = dataSpilt(70, TrainData)
    Val_Data = tempObject[1]
    TrainData1 = tempObject[0]

    # Split  Training Labels into Train Labels and Validation Labels
    tempObject = None
    tempObject = dataSpilt(70, TrainLabelsList)
    Val_Labels = tempObject[1]
    TrainLabels1 = tempObject[0]

    # Train Model
    clf = trainSVM(np.array(Features_selected, int), TrainData1, Val_Data, Val_Labels)

    # Predict Labels
    predict(TestData, clf)

    print('Run Time: ', (timeit.default_timer()) - startTime)  


