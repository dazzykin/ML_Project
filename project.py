#Group: Iskandar Askarov, Ketki Ambekar
#UCID: ia253, kpa9
#Email: ia253@njit.edu, kpa9@njit.edu
#Written in Python 3.6
#Expected Runtime of project: 
#Arguments for this program: 
#1) Training Data.gz
#2) Test Data.gz
#3) Training labels.txt

import sys
import traceback
import timeit

# #Read Args 
# file1_train = sys.argv[0]
# file2_test = sys.argv[1]
# file3_trainLabels = sys.argv[2]


#Read Args 
file1_train = sys.argv[1]
file2_test = sys.argv[2]
file3_trainLabels = sys.argv[3]

#Global Variables
startTime = timeit.default_timer()
TrainData = []
TestData = []
TrainLabels ={}
EXIT_FAILURE=0
EXIT_SUCCESS=1


#Step 1: Read Data. Train -> List , Test -> List, Labels (=target) -> dict
def readList(fileName=""):
    returnList=[]
    i=0
    rowVec=[]
    nom=0
    tempFile=None
    try:
        tempFile = (open(fileName, 'r')).read()    #open file in read only mode
        #Recording number of rows
        #No_of_rows=len(tempFile.splitlines())

        for row in tempFile.splitlines():  
            rowVec=[]
            #rowVec= [float(item) for item in row.split()] 
            for item in row.split():
                rowVec.append( float(item))
            
            # Recording highest no. of columns for ther purpose of knowing how many random weights to choose later. 
            # if len(rowVec) > No_of_columns:
            # No_of_columns   = len(rowVec)
            returnList.append(rowVec)     
    except:
        print("Unable to open {} File".format(fileName));
        #traceback.print_exc();
        print("Exiting");
        sys.exit(EXIT_FAILURE);
    finally:
        a=1
        # if(tempFile):
        #     tempFile.close();
    return returnList

def readDict(fileName=""):
    tempFile = None;
    returnDict = dict();

    try:
        #tempFile = (open(fileName, 'r')).read() 
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
        if(tempFile):
            tempFile.close();
        
    return returnDict;

#Step 2: Split Train Data into 2 Subsets (row wise) -> "train" (70%) and "Validation" (30%)

#Step 3: Calc Correlation of each column w.r.t the "target" ie the labels

#Step 4: Arrange the correlations in Descending order of absolute magnitude. 

#Step 5: Extract the 20 Columns

#Step 6: Train your SVM on extracted features

#Step 7: Predict for 

# Main Function 
if __name__ == '__main__':

    #List all files 
    # from os import listdir
    # print(str(listdir("/Users/ketkiambekar/Documents/")))

    #Read Files
    TrainLabels = readDict(file3_trainLabels)
    TrainData = readList(file1_train)
    TestData = readList(file2_test)



     
    print('Run Time: ', (timeit.default_timer()) - startTime)  
    

