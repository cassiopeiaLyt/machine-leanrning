# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def loadData(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        dataMat.append(curLine)
    return dataMat

# theta表达式求解
def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0:
        print "Thie matrix is singular, cannot do inverse"
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

if __name__== '__main__':
    (xArr, yArr) = loadDataSet('ex0.txt')
    print xArr
    print yArr
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr,yArr)
    yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    ax.plot(xMat[:, 1], yHat)
    plt.show()



