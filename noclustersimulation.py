import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance
import os
import pandas as pd 
import math
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn import preprocessing
import time
import random
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score
from collections import defaultdict
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
#preprocessing
#buat evaluasi menggunakan matrix clustering!!!!!!!!!!!!

def countcoordinatevelocity(xorycenters1,xorycenters2,length):
    return (xorycenters1 - xorycenters2) / length

def countvelocity2(xcenters, ycenters,prevVx,prevVy):#ambil data yang lama untuk diakumulasikan
    x = countcoordinatevelocity(xcenters[0], xcenters[-1],len(xcenters) )
    y = countcoordinatevelocity(ycenters[0], ycenters[-1], len(ycenters))
   # xlenin = (prevVx+(xcenters[-2] - xcenters[-1])/2)
   # ylenin = (prevVy+(ycenters[-2] - ycenters[-1])/2)
    xlen = (xcenters[-2] - xcenters[-1])
    ylen = (ycenters[-2] - ycenters[-1]) 
    if (math.isnan(xlen)):
        xlen = 0
    if (math.isnan(ylen)):
        ylen = 0
    
    #return math.sqrt((x** 2) + (y** 2)),xlen,ylen
    return calculate_direction(xlen,ylen),xlen,ylen

def calculate_direction(delta_x,delta_y):
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = (angle_degrees + 360) % 360  # Adjust to 0-360 degrees range
    return angle_degrees

def countvelocity(xcenters, ycenters):#ambil data yang lama untuk diakumulasikan
    x = countcoordinatevelocity(xcenters[0], xcenters[-1], len(xcenters))
    y = countcoordinatevelocity(ycenters[0], ycenters[-1], len(ycenters))
    xlen = xcenters[-2] - xcenters[-1]
    ylen = ycenters[-2] - ycenters[-1]
    if (math.isnan(xlen)):
        xlen = 0
    if (math.isnan(ylen)):
        ylen = 0
    return math.sqrt((x ** 2) + (y ** 2)), xlen, ylen
    
def countvectordirection(x1, x2, y1, y2):
    x = x1 - x2
    y = y1 - y2
    if(x==0):
        return 0
    res = math.degrees(math.atan(y / x))
    if (math.isnan(res)):
        res = 0
    return res
def smallest_angular_distance(angle1, angle2):
    # Calculate the absolute difference
    delta = abs(angle1 - angle2)
    
    # Normalize the difference
    delta = delta % 360
    
    # Adjust to get the smallest angle
    if delta > 180:
        delta = 360 - delta
    
    return delta
def getdatabyframe(df,frameno):
    dataframe = []
    for data in df:
        if (data[0] == frameno):
            dataframe.append(data)
    return dataframe
def getdatacentroidframe(centros, frameno):
    try:
        centroidframe = []
        for centro in centros:
            if (centro[0][0] == frameno):
                centroidframe.append(centro)
                break
    except Exception as ex:
        print(ex)    
    return centroidframe[0]

def getdatabynframe(df, startframe, endframe):
    dataframe = []
    
    for data in df:
        if (data[0] >= startframe and data[0]<endframe):
            dataframe.append(data)
    
    #temp= np.split(df, np.where(np.diff(df[:,1]))[0]+1)
    return dataframe
def reshapeperidped(data):
    uniq,junk = np.unique(df[:,1], 1)
    temp = []
    for idped in uniq:
        a = []
        a.append(idped)
        a.append(data[np.where(data[:, 1]== idped)])
        temp.append(a)
    return temp

def arrangepercentroid(centroids):
    centro=[]
    centrotemp=[]
    for id,eachframe in enumerate(centroids):
        for data in eachframe:
            if(len(centrotemp)==0):
                centro.append([data[1],[[data[0],data[2],data[3],data[4],data[5],data[6],data[7],data[8]]]])
                centrotemp.append(data[1])
            else:
                find=np.where(data[1]==np.array(centrotemp))[0]
                if(len(find)>0):
                    centro[find[0]][1].append([data[0],data[2],data[3],data[4],data[5],data[6],data[7],data[8]])
                else:
                    centro.append([data[1],[[data[0],data[2],data[3],data[4],data[5],data[6],data[7],data[8]]]])
                    centrotemp.append(data[1])
    #centro id, frame, x,y,degree,directionvectorx,directionvectory, 
    return centro
directory='crossing_90_g_3_outputfull simule.txt'
#directory='Pedestrian Cluster/dataset continual/crossing_120_b_02output ready.txt'
def preparethedata(directory):
    #df = pd.read_csv(directory, dtype={'frame_num':'int','ped_id':'int' }, delimiter = ' ',  header=None,names=['frame_no','peds_id','xcenter','ycenter','V','Vx','Vy','radius','jumlahcluster'])
    df = pd.read_csv(directory, dtype={'frame_num':'int','ped_id':'int' }, delimiter = ' ',  header=None,names=['frame_no','peds_id','xcenter','ycenter','V','Vx','Vy','radius','jumlahcluster'])
    #generate velocity and direction
    df.insert(4, "va", 0, True)
    df.insert(5, "vx", 0, True)
    df.insert(6,"vy",0,True)
    df.insert(7, "direction", 0, True)

    i = 0
    before=-1
    xcenters = []
    ycenters = []
    df=df.to_numpy()
    databefore = -1
    prevVx = 0
    prevVy = 0
    #ini harus dicari sih datanya dari pada mengandalkan susunan data awal
    #data disusun berdasarkan id pedestrian, jika Id pedestriannya berubah dari sebelumnya maka di set 0 nilainya
    for data in df:
        if (i > 0):
            if(df[before][1] != data[1]):
                xcenters = []
                ycenters = []
                i = 0
                prevVx = 0
                prevVy = 0
            else:
                a=0
        xcenters.append(data[2])
        ycenters.append(data[3])
        if (i > 0):
            #databefore=df[i][1]
            if (i > 100):
                a=0
            if((data[0]==122 or data[0]==121)and data[1]==793):
                a=0
            data[4], data[5], data[6] = countvelocity2(xcenters, ycenters, prevVx, prevVy)
            #data[4], data[5], data[6] = countvelocity(xcenters, ycenters)
            prevVx = data[5]
            prevVy = data[6]
            if (math.isnan(data[5])):
                a=0
            data[7]= countvectordirection(xcenters[-2],xcenters[-1],ycenters[-2],ycenters[-1])
        i += 1
        before += 1
    return df
#data berisi: frame,idpeds,x,y,degree,dvx,dvy,clusterid

df=preparethedata(directory)
start=900
finish=2200
def direction_vectors(trajectory):
    vectors = []
    for i in range(len(trajectory) - 1):
        p1 = np.array(trajectory[i])
        p2 = np.array(trajectory[i + 1])
        vector = p2 - p1
        if(vector[0]==0 and vector[1]==0):
            vectors.append(np.array([0,0]))
        else:    
            vectors.append(vector / np.linalg.norm(vector))
    return vectors
#arrange=arrangeperperson(dataGT)
def dtw_normalized_vectors(trajectory1, trajectory2):
    vectors1 = direction_vectors(trajectory1)
    vectors2 = direction_vectors(trajectory2)
    if(len(vectors1)==0 or len(vectors2)==0):
        distance=999
    #    print('vector kosong')
    else:
        distance, _ = fastdtw(vectors1, vectors2, dist=euclidean)
    return distance
def compareTrajectorysimilaritywithGT(arrange,evsmooth):
    #convertdataGT perperson
    centroidsimvals=[]
    countpeds=0
    for idarr,gt in enumerate(arrange):
        simval=[]
        countcluster=0
       # if(len(np.array(gt[1])[:,(1,2)])==1):
       #     print('peds',gt[0])
        for idev,Evs in enumerate(evsmooth):
         #   if(len(np.array(Evs[1])[:,(4,5)])==1):
         #       print('clust',Evs[0])
            #data1=np.array(Evs[1])[:,(1,2)]
            #data2=np.array(gt[1])[:,(1,2)]
            
            similarityval=dtw_normalized_vectors(np.array(Evs[1])[:,(4,5)],np.array(gt[1])[:,(1,2)])
                
        
            if(similarityval<4):
                simval.append([Evs[0],similarityval])
                print('similval:',similarityval)
                
                countcluster+=1
                #break
        centroidsimvals.append([gt[0],simval])
        if(countcluster>=1):
            countpeds+=1
            print('countpeds:',countpeds)
    percentage=countpeds/len(arrange)
    print('peds=',countpeds)

    return centroidsimvals,percentage
datas = getdatabyframe(df, start)
datanframe = getdatabynframe(df, start, finish)
datasn = np.array(datanframe)
def normalize_dataset(datasn):
    return datasn

datas=np.array(datas)
datafit = datas[:, [4]]
datafit=datafit.reshape(-1,1)
bandwidth = estimate_bandwidth(datafit, quantile=0.2, n_samples=100)

idp=np.array(datas[:,1])
xpoints = datas[:,2]
ypoints = datas[:,3]
##################make plotss############################
#lakukan evaluasi dengan mengubah semua cluster denan cluster terakhir yang dimiliki oleh suatu pedestrian harusnya jalannya akan lebih stabil sebagai input
#lalu bagaimana dengan saat inference bisa saja langsung diinputkan, mungkin perlu ditambah denan sedikit evaluasi untuk
############################animasi plot##################################
figA, axA = plt.subplots()
annotation=[]
for i in range(len(xpoints)):
    ann = str(idp[i])
   # annotation.append([labelsK[i][0],pedid[i],axA.annotate(ann, xy=(xpoints[i],ypoints[i]), xytext=(xpoints[i],ypoints[i]))])
    annotation.append([ann,ann,axA.annotate(ann, xy=(xpoints[i],ypoints[i]), xytext=(xpoints[i],ypoints[i]))])
scat = axA.scatter(xpoints, ypoints, s=10)
###############################plot pertama plot sublot x=cluster y=nilai per plotnya adalah frame#############################
def idxreturnpedestrian(anno,personID):
    #mencari posisi yang sesuai dengan annotation
    return np.where(personID == anno)
def update(frame):
    frame=frame+start
    dataA = getdatabyframe(datanframe,frame)
    datasupdate = np.array(dataA)
    
    #datas=normalize_dataset(datas)
    xpointsA = [sublist[2] for sublist in datasupdate]
    ypointsA = [sublist[3] for sublist in datasupdate]
    personID = [sublist[1] for sublist in datasupdate]

    #numberN = [sublist[-1] for sublist in datasupdate]

    try:
        # for idx, d in enumerate(datasupdate):#ini untuk mencari yang label pada data updatesnya masih nan akan ditampilkan sebagai -1
        #     convert = np.array(annotation)[:, 0]
        #     foundya = np.where(convert == str(d[1]))
        #     if (len(foundya[0]) == 0):
        #         lblnew=str(str(d[1]))
        #         annotation.append([str(d[1]), lblnew, axA.annotate(lblnew, xy=(500, 500), xytext=(500, 500))])

        for num, annot in enumerate(annotation):
            idx=idxreturnpedestrian(float(annot[0]),personID)[0]#get the id from annotation#ganti disini ya buat nampilin
           # print(personID)

            if (len(idx)> 0):
                annot[2].set_position((xpointsA[num], ypointsA[num]))  #coba ubah annotation cluster biar kelihatan cuk
                labeldirvel=str(personID[num])#+ '/' + "{:.2f}".format(directX[idx[0]])+'-'+ "{:.2f}".format(directY[idx[0]])
                annot[2].set_text(labeldirvel)#str(labelsK[idx[0]][0])+"/"+str(labelidx[0][0]) )
            else:
                annot[2].set_text(" ")
               # print("empty")

    except Exception as e:
        print('no 9 ',e)
        a=9
    # for a in labelsc:
    #     labelsMark.append(markerdict[int(a)])
    
    dataAn = np.stack([xpointsA, ypointsA]).T
    scat.set_offsets(dataAn)
    #scat.set_color(labelscolors1)
    return (scat,annotation)


#centroids.append([clus[0][0], clus[0][-1], x, y, V, Vx, Vy])  #frame,idcluster,x,y,V,Vx,Vy


ani = animation.FuncAnimation(fig=figA, func=update, frames=1000, interval=100)

plt.show()      
#buat clusteringnya cuys


# Background color