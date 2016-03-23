#-*- coding:utf8 -*-
#coding=utf-8
from math import log
import operator
import treePlotter
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec = featVec[axis+1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #特征的数量
    baseEntropy = calcShannonEnt(dataSet)  #整个数据集的原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]    #取出第i个特征中所有的值
        uniqueVals = set(featList)                        #将第i个特征中同类的值合并
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #以第i个特征中的value值进行分类
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  #计算出分类后子类的香农熵，然后乘以对应的prob,再求和
        infoGain = baseEntropy - newEntropy        #更新和原始香农熵差别最大的香农熵
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=1)
    print sortedClassCount
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):   #类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:                  #表示特征遍历完了，每遍历一个特征就删除一个，最后只剩下一个分类标签
        return majorityCnt(classList)         #特征处理完了，但是类标签不唯一，返回出现次数最多的类
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]          #获得最好的分类对应的标签
    myTree = {bestFeatLabel:{}}               #用字典构建树
    del(labels[bestFeat])                     #移除最好的分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVac):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVac[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVac)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):  #将决策树以文件的形式保存在硬盘中，文件名为filename
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):  #将决策树从文件中打开,返回一个dict
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# myDat, labels = createDataSet()
#
myTree = treePlotter.retrieveTree(0)
# print classify(myTree, labels, [1, 0])

storeTree(myTree, 'classifierStorage.txt')
pp = grabTree('classifierStorage.txt')