import csv
import numpy as np
import pandas as pd

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

LDAllLncDisease = []
ReadMyCsv(LDAllLncDisease, '../pi_dis.csv')

AllLnc = []
ReadMyCsv(AllLnc, '../piRNA.csv')

AllDisease = []
ReadMyCsv(AllDisease, '../dis.csv')

AssociationMatrix = np.zeros((len(AllLnc), len(AllDisease)), dtype=int)

counter = 0
while counter < len(LDAllLncDisease):
    lnc = LDAllLncDisease[counter][0]
    disease = LDAllLncDisease[counter][1]

    flag1 = 0
    counter1 = 0
    while counter1 < len(AllLnc):
        if lnc == AllLnc[counter1][0]:
            flag1 = 1
            break
        counter1 = counter1 + 1

    flag2 = 0
    counter2 = 0
    while counter2 < len(AllDisease):
        if disease == AllDisease[counter2][0]:
            flag2 = 1
            break
        counter2 = counter2 + 1

    if flag1 == 1 and flag2 == 1:
        AssociationMatrix[counter1][counter2] = 1

    counter = counter + 1

StorFile(AssociationMatrix, 'AssociationMatrix.csv')

print(np.array(AssociationMatrix).shape)
