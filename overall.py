    # -*- coding: utf-8 -*-
import numpy
import random
import codecs
import copy
import re
import matplotlib.pyplot as plt
import project
import sys
import numpy as np
from numpy import mean
from numpy import cov
from numpy import linalg
from numpy import mat
from numpy import argsort



#IPCA降维
def IPCA(data,N):
# Normalise data #
	data = np.array(data)
	Norm = preprocessing.Normalizer()#转化为布尔值，用于概率估计
	Norm.fit(data)#训练
	data = Norm.transform(data)
# IncrementalPCA model #
	ipca = IncrementalPCA(n_components=N)#设定维数
	ipca.fit(data)#训练

	length = len(data)
	chunk_size = 6#设定模块数
	pca_data = np.zeros(shape=(length, ipca.n_components))

	for i in range(0, length // chunk_size):
   		 ipca.partial_fit(data[i*chunk_size : (i+1)*chunk_size])
   		 pca_data[i * chunk_size: (i + 1) * chunk_size] = ipca.transform(data[i*chunk_size : (i+1)*chunk_size])
	new_test_data = ipca.transform(data)
	#print(new_test_data)
	return new_test_data
    
def pca(dataMat, K=3000):   # dataMat是原始数据，一个矩阵，K是要降到的维数  
    meanVals = mean(dataMat, axis=0)   # 第一步:求均值  
    meanRemoved = dataMat - meanVals    # 减去对应的均值  
  
    covMat = cov(meanRemoved, rowvar=0)   # 第二步,求特征协方差矩阵  
  
    eigVals, eigVects = linalg.eig(mat(covMat))   # 第三步,求特征值和特征向量  
    eigValInd = argsort(eigVals)   # 第四步,将特征值按照从小到大的顺序排序  
    eigValInd = eigValInd[: -(K+1): -1]  # 选择其中最大的K个  
    redEigVects = eigVects[:, eigValInd]       # 然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵.  
  
    lowDDataMat = meanRemoved * redEigVects   # 第五步,将样本点投影到选取的特征向量上,得到降维后的数据  
  
    reconMat = (lowDDataMat * redEigVects.T) + meanVals   # 还原数据  
    #contribution = calc_single_contribute(eigVals, eigValInd)   # 计算单维贡献度,总贡献度为其和  
    return lowDDataMat


def calcuDistance(vec1, vec2):
    # 计算向量vec1和向量vec2之间的欧氏距离
    return numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))

def getFilelist(argv) :
    path = argv
    filelist = []
    files = os.listdir(path)
    for f in files :
        if(f[0] == '.') :
            pass
        else :
            filelist.append(f)
    return filelist,path

def loadDataSet(inFile):
	# 载入数据测试数据集
    # 数据由文本保存，为二维坐标
    inDate = codecs.open(inFile, 'rb', 'utf-8').readlines() 
    dataSet = list()
    for line in inDate:
    	line = line.strip()
    	strList = re.split('[ ]+', line)  # 去除多余的空格
    	# print strList[0], strList[1]
    	numList = list()
    	for item in strList:
    		num = float(item)
    		numList.append(num)
    		# print numList
    	dataSet.append(numList)

    return dataSet      # dataSet = [[], [], [], ...]

def initCentroids(dataSet, k):
	# 初始化k个质心，质心为文档最后几篇，即为新建的8篇关键词文档
        init_list=list()
        list_length=len(dataSet)
        for item in dataSet[list_length-k:list_length]:
            init_list.append(item)
        return init_list
	#return random.sample(dataSet, k)  # 从dataSet中随机获取k个数据项返回

def minDistance(dataSet, centroidList):
    # 对每个属于dataSet的item，计算item与centroidList中k个质心的欧式距离，找出距离最小的，
    # 并将item加入相应的簇类中
    init_list=list()
    list_length=len(dataSet)
    clusterDict = dict()                 # 用dict来保存簇类结果
    for item in dataSet[:list_length]:
        vec1 = numpy.array(item[1:])         # 将Dataset中标号以后的数据转换成array形式，用于计算miniDistance
        flag = 0            # 簇分类标记，记录与相应簇距离最近的那个簇
        minDis = float("inf")               # 初始化为最大值
	
        for i in range(len(centroidList)):
            vec2 = numpy.array(centroidList[i][1:])#将质心点列表的标号以后作为质心点向量
            distance = calcuDistance(vec1, vec2)  # 计算相应的欧式距离
            if distance < minDis:    
                minDis = distance
                flag = i                          # 循环结束时，flag保存的是与当前item距离最近的那个簇标记
		
        if flag not in clusterDict.keys():   # 簇标记不存在，进行初始化
            clusterDict[flag] = list()
    	# print flag, item
        clusterDict[flag].append(item)       # 加入相应的类别中

    return clusterDict                       # 返回新的聚类结果

def getCentroids(clusterDict):
    # 得到k个质心
    centroidList = list()
    for key in clusterDict.keys():
        A=numpy.array(clusterDict[key])
        #若矩阵A的列的维度大于给定维度+1，则减一，实际为：去掉不参与计算第一列的标号
        A=numpy.delete(A,0,1)    
        centroid = numpy.mean(A, axis = 0)# 计算每列的均值，即找到质心
        centroid = numpy.insert(centroid,0,0)
        # print key, centroid
        centroidList.append(centroid)
    
    return numpy.array(centroidList).tolist()

def getVar(clusterDict, centroidList):
    # 计算簇集合间的均方误差
    # 将簇类中各个向量与质心的距离进行累加求和

    sum = 0.0
    for key in clusterDict.keys():
        vec1 = numpy.array(centroidList[key][1:])#将中心点列表标号以后元素作为中心点向量
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = numpy.array(item[1:])#将中心点列表标号以后元素作为中心点向量
            distance += calcuDistance(vec1, vec2)
        sum += distance

    return sum

def showCluster(centroidList, clusterDict):
    # 展示聚类结果

    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']      # 不同簇类的标记 'or' --> 'o'代表圆，'r'代表red，'b':blue
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']   # 质心标记 同上'd'代表棱形
    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize = 12)  # 画质心点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key]) # 画簇类下的点

    plt.show()

if __name__ == '__main__':
    #文本预处理、分词
    (allfile,path) = project.getFilelist("C:/Users/56486/Desktop/test_english/")
    for ff in allfile :
        print("Using jieba on "+ff)
        project.fenci(ff,path)

    project.Tfidf(allfile)
        
    inFile = r"C:/Users/56486/AppData/Local/Programs/Python/Python36-32/tfidffile/root.txt"           # 数据集文件 
    dataSet = loadDataSet(inFile)                      # 载入数据集
    #dataSet=pca(dataSet, 500)
    centroidList = initCentroids(dataSet, 8)            # 初始化质心，设置k=8
    clusterDict = minDistance(dataSet, centroidList)   # 第一次聚类迭代
    newVar = getVar(clusterDict, centroidList)         # 获得均方误差值，通过新旧均方误差来获得迭代终止条件
    oldVar = -0.000001                                   # 旧均方误差值初始化为-1
    print( '***** 第1次迭代 *****')
    print(' ')
    print('簇类')
    for key in clusterDict.keys():
        #print(key, ' --> ', clusterDict[key])
        for item in clusterDict[key]:
            print (key, ' --> ', item[0])
            
    #print('k个均值向量: ', centroidList)
    #print('平均均方误差: ', newVar)
    print(' ') 
    #showCluster(centroidList, clusterDict)             # 展示聚类结果

    k = 2
    while abs(newVar - oldVar) >= 0.000001:              # 当连续两次聚类结果小于0.0001时，迭代结束          
        centroidList = getCentroids(clusterDict)
        #print(centroidList)# 获得新的质心
        clusterDict = minDistance(dataSet, centroidList)  # 新的聚类结果
        oldVar = newVar                                   
        newVar = getVar(clusterDict, centroidList)

        print ('***** 第%d次迭代 *****' % k)
        print (' ')
        print ('簇类')
        for key in clusterDict.keys():
            #print(key, ' --> ', clusterDict[key])
            for item in clusterDict[key]:
                #print(key)
                print (key, ' --> ', item[0])
            
        #print ('k个均值向量: ', centroidList)
        #print ('平均均方误差: ', newVar)
        print (' ')
        #showCluster(centroidList, clusterDict)            # 展示聚类结果
        k += 1
