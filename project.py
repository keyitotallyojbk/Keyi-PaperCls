# -*- coding: utf-8 -*-
import os
import jieba
import jieba.posseg as pseg
import sys
import string
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import *

#英文合并同类词
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


#获取文件列表（该目录下放着100份文档）
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

#对文档进行分词处理
def fenci(argv,path) :
    #保存分词结果的目录
    sFilePath = './segfile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)
    #读取文档
    filename = argv
    f = open(path+filename,'r+',encoding='utf-8')
    file_list = f.read()
    f.close()
    
    #对文档进行分词处理，采用默认模式
    seg_list = jieba.cut(file_list,cut_all=True)
  
    #中文停词
    result=[]
    stopwords_chi = [line.strip() for line in open(r'C:\Users\56486\Desktop\Textfile\stopwords.txt').readlines()]
    for seg in seg_list:
        if seg=='摘要':
            break
        if seg not in stopwords_chi:
            if (seg != '' and seg != "\n" and seg != "\n\n"):
                seg = ''.join(seg.split())
                result.append(seg)
    #将分词后的结果用空格隔开，保存至本地。比如"我来到北京清华大学"，分词结果写入为："我 来到 北京 清华大学"
    f = open(sFilePath+"/"+filename,"w+")
    f.write(' '.join(result))
    f.close()

#读取100份已分词好的文档，进行TF-IDF计算
def Tfidf(filelist):
    path = './segfile/'
    corpus = []  #存取100份文档的分词结果
    for ff in filelist :
        fname = path + ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)    

    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    
    word1 = vectorizer.get_feature_names() #所有文本的关键字
    word2=list()
    for item in word1:
        if item not in stopwords.words('english'):
            word2.append(item)
    #英文去模糊词    
    stemmer = PorterStemmer()
    stemmed = stem_tokens(word2, stemmer)
    word=list()
    for item in stemmed:
        word.append(item)
    weight = tfidf.toarray()              #对应的tfidf矩阵
    
    sFilePath = './tfidffile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)

    text=open(sFilePath+'/'+'root.txt','w+')
    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)) :
        print(u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+str(i).zfill(5)+'.txt',"--------")
        f = open(sFilePath+'/'+str(i).zfill(5)+'.txt','w+')
        text.write(str(i)+' ')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n\n")
            text.write(str(weight[i][j])+"  ")
        text.write("\n")
        f.close()
    text.close() 
        
if __name__ == "__main__" : 
    (allfile,path) = getFilelist("C:/Users/56486/Desktop/test_chinese")
    for ff in allfile :
        print("Using jieba on "+ff)
        fenci(ff,path)

    Tfidf(allfile)
