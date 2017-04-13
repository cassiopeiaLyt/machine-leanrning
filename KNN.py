# coding=utf-8
from numpy import *
import operator

# 创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# 10%作为训练集 90%作为测试集
# inX为输入的分类向量
def classify0(inX,dataSet,labels,k):
    dataSetSize =  dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    # axis=0 为普通相加 axis=1为行向量相加
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    # argsort()返回数值从小到大索引
    sortedDistancies = distances.argsort()
    classCount = {}
    for i in range(k):
        votedLabel = labels[sortedDistancies[i]]
        classCount[votedLabel] = classCount.get(votedLabel,0) +1
    # itemgetter用于获取数据哪一维的数据
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 约会网站数据分析
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 截取所有回车字符
        line = line.strip()
        # 得到分割的数组
        listFromLine = line.split('\t')
        # 选取前三列放入矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # 索引-1表示最后一列，放入分类标签矩阵中
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

# 归一化
# newValue = (oldValue - min)/(max - min)
def autoNorm(dataSet):
    # 参数0表示从列中选取最小值和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals -minVals
    # 创造和样本集一样大小的零矩阵
    # shape 返回行数和列数
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # tile用于重复数组
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges,minVals

# 测试函数
def datingClassTest():
    # 选取多少数据测试分类器
    hoRatio = 0.10
    datingDataMat,datingDataLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #设置测试个数
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingDataLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingDataLabels[i])
        if (classifierResult != datingDataLabels[i]) : errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))
    print errorCount

def classifyPerson():
    resultList = ['not at all','in small does','in large does']
    percentTats = float(raw_input("percentage of time spend playing vedio games?"))
    ffMiles = float(raw_input("frequent filter miles earns per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    (datingDataMat,datingLabels) = file2matrix("datingTestSet2.txt")
    (normMat,ranges,minVals) = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print "you would probably like this person:",resultList[classifierResult-1]


if __name__== '__main__':
    '''
    k=3
    group,labels = createDataSet()
    output = classify0(test,group,labels,k)
    print "input:",test,"classified result:",output

    test = array([1, 1])
    group, labels = createDataSet()
    output = classify0(test, group, labels, k)
    print "input:", test, "classified result:", output
    '''
    # datingClassTest()
    classifyPerson()

