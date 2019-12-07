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

def dataSpilt(part1_percent=0.0, dataSet=np.array):
    end = len(dataSet)
    mid = end * (part1_percent / 100)
    return1 = np.array()
    return2 = np.array()

    try:
        for idx in (0, mid, 1):
            return1 = np.append(return1, dataSet[idx])
    except:
        a = 1
    finally:
        a = 1


# Step 3: Calc Correlation of each column w.r.t the "target" ie the labels
def getPearsonCorrelation(List1, List2):
    corr = 0.0
    corr = (getSpread(List1) * getSpread(List2)) / (getSD(List1) * getSD(List2))
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


def calcCorrelations(dataSet=np.array):
    scores = []

    for j in np.transpose(dataSet):
        scores.append(getPearsonCorrelation(dataSet[:, j]), TrainLabels)

    scores.sort(reverse=True)
    return scores


# Step 4: Arrange the correlations in Descending order of absolute magnitude.


def getTopFeatures(No_of_features=0, feature_ranks=[]):
    a = 1


# Step 5: Extract the top 20 Columns

# Step 6: Train your SVM on extracted features

# Step 7: Predict for test data

# Main Function 
if __name__ == '__main__':
    # List all files
    # from os import listdir
    # print(str(listdir("/Users/ketkiambekar/Documents/")))

    # Read Files
    TrainLabels = readDict(file3_trainLabels)
    TrainData = readList_np(file1_train)
    TestData = readList(file2_test)

    TrainLabelsList = convertDict2List(TrainLabels)

    print('Run Time: ', (timeit.default_timer()) - startTime)  


