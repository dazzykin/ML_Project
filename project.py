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
import random
import numpy as np
from sklearn import svm;
from sklearn import model_selection
import os;
import traceback;

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
TrainLabels = None
# TrainLabelsList=[]
Val_Data = []
TrainData1 = None
No_of_rows = None
No_of_columns = None

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


def shuffle(avFeatures, avLables):
    mapper = list(zip(avLables, avFeatures));
    random.shuffle(mapper);
    training_limit = int(len(mapper) * VALIDATION_SPLIT);
    avLables, avFeatures = zip(*mapper);

    avLables = np.array(avLables);
    avFeatures = np.array(avFeatures);

    return avLables[:training_limit], avFeatures[:training_limit], avLables[training_limit:], avFeatures[
                                                                                              training_limit:];


@importer
def import_data(fp):
    data = [];
    for row in fp:
        data.append(convert_to_int(row.split()));
    return data;


# Step 1: Read Data. Train -> List , Test -> List, Labels (=target) -> dict
def readList(fileName=""):
    returnList = []
    i = 0
    rowVec = []
    nom = 0
    tempFile = None
    global No_of_rows
    global No_of_columns
    try:
        tempFile = (open(fileName, 'r')).read()  # open file in read only mode
        # Recording number of rows
        No_of_rows = len(tempFile.splitlines())

        for row in tempFile.splitlines():
            rowVec = []
            # rowVec= [float(item) for item in row.split()]
            for item in row.split():
                rowVec.append(float(item))

            # Recording highest no. of columns for ther purpose of knowing how many random weights to choose later.
            if len(rowVec) > No_of_columns:
                No_of_columns = len(rowVec)

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
        returnList = np.loadtxt(fileName, dtype=np.int8)

    except:
        print("Unable to open {} data".format(fileName));
        traceback.print_exc();
        print("Exiting");
        sys.exit(EXIT_FAILURE);
    finally:
        a = 1
    return returnList

    # def readDict(fileName=""):
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


# def convertDict2List(my_dict={}):
#     returnList=[]

#     for k,v in my_dict.items():
#         returnList.append([k,v])

#     return returnList

# Step 2: Split Train Data into 2 Subsets (row wise) -> "train" (70%) and "Validation" (30%)
def dataSpilt(part1_percent=0.0, trainDataSet=np.array, trainLabels=np.array):
    end = len(trainDataSet)
    mid = int(end * (part1_percent / 100))

    return_trainData = np.array([], dtype=np.int8)
    return_valData = np.array([], dtype=np.int8)
    return_trainLabels = np.array([], dtype=np.int8)
    return_valLabels = np.array([], dtype=np.int8)

    tempRowNums_train = []
    tempRowNums_val = []

    try:

        # Randomize the split
        while len(tempRowNums_train) < mid:
            a = random.randint(0, end)
            if not a in tempRowNums_train:
                tempRowNums_train.append(a)

        # Get train data, Labels
        for k in tempRowNums_train:
            np.append(return_trainData, trainDataSet[k,])
            return_trainData.append(trainDataSet[k,])
            return_trainLabels.append(trainLabels[k,])

        # Get val Data, Labels
        for k in range(0, end, 1):
            if not k in tempRowNums_train:
                return_valData.append(trainDataSet[k,])
                return_valLabels.append(trainLabels[k,])

    except:
        a = 1
    finally:
        a = 1
    return return_trainData, return_trainLabels, return_valData, return_valLabels


# Step 3: Calc Correlation of each column w.r.t the "target" ie the labels and arrange in Descending order of absolute magnitude.
def getPearsonCorrelation(List1=np.array, List2=np.array):
    corr = 0.0

    spread1 = getCovariance(List1)
    spread2 = getCovariance(List2)

    stdDev1 = getSD(List1)
    print("SD1: " + str(stdDev1))
    stdDev2 = getSD(List2)
    print("SD2: " + str(stdDev2))

    corr = (spread1 * spread2) / (stdDev1 * stdDev2)
    corr = abs(corr)
    print("My Correlation: {}".format(corr))
    #  print("Numpy Corr:"+str(float(np.corrcoef(List1,List2))))
    return corr


def getCovariance(List1=np.array):
    mean = sum(List1) / len(List1)
    returnVal = 0.0
    for k in List1:
        returnVal = returnVal + (k - mean)

    return (returnVal)


def getSD(List1=[]):
    mean = sum(List1) / len(List1)
    returnVal = 0.0
    for k in List1:
        # print("SD: "+str(returnVal+((k - mean)**2)))
        returnVal = returnVal + ((k - mean) ** 2)

    return (returnVal ** 0.5)


def calcCorrelations(dataSet1=np.array, targetLabels1=np.array):
    scores = []
    # print(str(dataSet)))
    # print(str(np.transpose(dataSet)))
    dataSet = np.array(dataSet1, dtype=np.int8)
    targetLabels = np.array(targetLabels1, dtype=np.int8)

    # for j in np.transpose(dataSet):
    for j in range(0, len(dataSet[0]), 1):
        print(str(dataSet[:, j]))
        print(type(dataSet[:, j]))
        print(str(targetLabels))
        # print(type(targetLabels[:, 0]))

        corr = getPearsonCorrelation(dataSet[:, j], targetLabels)
        scores.append([corr, j])

    # scores.sort(reverse=True)
    scores_np = np.array(scores, dtype=np.int8)
    # Sort Scores array based one 0th Column
    scores_np_sorted = scores_np[scores_np[:, 0].argsort()]
    return scores_np_sorted


# Step 4: Extract the top x Columns
def getTopFeatures(No_of_features=0, feature_ranks=np.array):
    return feature_ranks[0:No_of_features, :0]


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
    # TrainLabels = readList_np(file3_trainLabels)
    # TrainData = readList_np(file1_train)
    # TestData = readList_np(file2_test)

    # Convert Test labels from dictonary to numpy array
    # TrainLabelsList=np.array(convertDict2List(TrainLabels),float)

    # #Calculate pearson Correlation of each column with target label data
    # Feature_ranking=calcCorrelations(TrainData, TrainLabels)

    # #Get top 20 Features
    # Features_selected= getTopFeatures(5,Feature_ranking)
    # print("Features Selected are: "+ str(Features_selected))

    # Split  training dataset into Train data and Validation Data
    # tempObject = dataSpilt(70, TrainData, TrainLabels)

    # isk
    # read data
    TrainData = import_data(file1_train)
    TrainLabels = import_labels(file3_trainLabels)
    TestData = import_data(file2_test)

    tempObject = shuffle(TrainData, TrainLabels)

    TrainData1 = tempObject[0]
    TrainLabels1 = tempObject[1]
    Val_Data = tempObject[2]
    Val_Labels = tempObject[3]

    # Calculate pearson Correlation of each column with target label data
    Feature_ranking = calcCorrelations(TrainData, TrainLabels)

    # Get top 20 Features
    Features_selected = getTopFeatures(20, Feature_ranking)
    print("Features Selected are: " + str(Features_selected))

    # Train Model
    # clf=trainSVM(np.array(Features_selected,int),TrainData1,Val_Data,Val_Labels)
    clf = trainSVM(TrainData, TrainData1, Val_Data, Val_Labels)

    # Predict Labels
    predict(TestData, clf)

    print('Run Time: ', (timeit.default_timer()) - startTime)  


