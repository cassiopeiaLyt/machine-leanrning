# coding=utf-8
# 使用随机梯度下降实现线性回归
# 单变量
import numpy as np
import matplotlib.pyplot as plt
import random

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

def bgd(xArr,yArr,alpha):
    size = len(xArr)
    n = len(xArr[0])
    theta = np.zeros(n)
    print theta
    for times in range(100):
       loss = np.dot(xArr,theta) -yArr
       gradient = np.dot(np.array(xArr).T,loss)/size
       theta = theta - alpha * gradient
       print times, theta
    return theta

def sgd(xArr,yArr,alpha):
    size = len(xArr)
    n = len(xArr[0])
    theta = np.zeros(n)
    data = []
    for i in range(size):
        data.append(i)
    xTrains = np.array(xArr).T # 转置，每一列代表一个样本点
    # 这里随机选择一个样本点进行更新（批量则需要全部考虑再除以n）
    for i in range(0,100):
        hypothesis = np.dot(xArr,theta)
        loss = hypothesis - yArr
        index = random.sample(data,1)
        cur = index[0]
        gradient = np.array(xArr[cur]) * loss[cur]
        theta = theta - alpha * gradient
        print i, theta
    return theta

# k为mini-batch的数量
def minibgd(xArr,yArr,alpha,k):
    size = len(xArr)
    n = len(xArr[0])
    theta = np.zeros(n)
    data = []
    for i in range(size):
        data.append(i)
    xTrains = np.array(xArr).T # 转置，每一列代表一个样本点
    # 这里随机选择一个样本点进行更新（批量则需要全部考虑再除以n）
    for i in range(0,100):
        hypothesis = np.dot(xArr,theta)
        loss = hypothesis - yArr
        index = random.sample(data,k)
        for cur in index:
            gradient = np.array(xArr[cur]) * loss[cur]
            theta = theta - alpha * gradient
        print i, theta
    return theta


if __name__== '__main__':

    (xArr, yArr) = loadDataSet('ex0.txt')
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    ws = minibgd(xArr,yArr,0.2,3)
    yHat = xMat * np.mat(ws).T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    ax.plot(xMat[:, 1], yHat)
    plt.show()
    '''
    test()
    '''








