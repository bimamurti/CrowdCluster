import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import cv2
from sklearn.cluster import AgglomerativeClustering
#from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance
import os
import pandas as pd 
import math
#from sklearn.metrics import silhouette_score
#from sklearn.metrics import calinski_harabasz_score
#from sklearn import preprocessing
#import time
#import random
import matplotlib.animation as animation
import matplotlib.colors as mcolors
#from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
#from sklearn.metrics import mutual_info_score, adjusted_rand_score
from collections import defaultdict
#from scipy.spatial.distance import euclidean


#preprocessing
#directoryGT='Pedestrian Cluster/crossing_90c03.txt'
#directoryGT='Pedestrian Cluster/dataset continual/MOT21GT02.txt'
directoryGT='gt03.txt'
#directory='Pedestrian Cluster/pred.txt'
directory='gt03.txt'
#directory='Pedestrian Cluster/crossing_90c03.txt'
#directory='Pedestrian Cluster/dataset continual/MOT21GT02.txt'
start=75#75
finish=950#2325#16fps 90 seconds
tdist=110
tdirect=50
filename = "MOT21GT02_outputfull.csv"
# tdist=50
# tdirect=40


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
    res = math.degrees(math.atan(y / x))
    if (math.isnan(res)):
        res = 0
    
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

def arrangeperperson(dataGT):
    persons=[]
    persontemp=[]
    for id,data in enumerate(dataGT):
        if(len(persontemp)==0):
            persons.append([data[1],[[data[0],data[2],data[3],data[4],data[5],data[6]]]])
            persontemp.append(data[1])
        else:
            find=np.where(data[1]==np.array(persontemp))[0]
            if(len(find)>0):
                persons[find[0]][1].append([data[0],data[2],data[3],data[4],data[5],data[6]])
            else:
                persons.append([data[1],[[data[0],data[2],data[3],data[4],data[5],data[6]]]])
                persontemp.append(data[1])
    return persons
            
def preparethedata(directory):
    df = pd.read_csv(directory, dtype={'frame_num':'int','ped_id':'int' }, delimiter = ' ',  header=None,names=['frame_no','peds_id','xcenter','ycenter'])
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
    #The data is arranged based on the pedestrian ID. If the pedestrian ID changes from the previous one, then its value is set to 0.
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
            data[7]= math.nan #countvectordirection(xcenters[-2],xcenters[-1],ycenters[-2],ycenters[-1])
        i += 1
        before += 1
    return df
#data contains: frame,idpeds,x,y,degree,dvx,dvy,clusterid

df=preparethedata(directory)
dfGT=preparethedata(directoryGT)

dataGT=getdatabynframe(dfGT, start, finish)


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

datas = getdatabyframe(df, start)
datanframe = getdatabynframe(df, start, finish)
arrange=arrangeperperson(dataGT)
#arrange2=arrangeperperson(datanframe)
#print('percentage:',compgt[1])
datasn = np.array(datanframe)
def normalize_dataset(datasn):
    # datasn[:, 2] = datasn[:, 2] / datasn[:, 2].max()
    # datasn[:, 3] = datasn[:, 3] / datasn[:, 3].max()
    #datasn[:, 4] = datasn[:, 4] / datasn[:, 4].max()
    # datasn[:, 5] = datasn[:, 5] / datasn[:, 5].max()
    # datasn[:, 6] = datasn[:, 6] / datasn[:, 6].max()
   # norms23 = np.linalg.norm(datasn[:,[2,3]], axis=1, keepdims=True)
   # norms56 = np.linalg.norm(datasn[:,[5,6]], axis=1, keepdims=True)
    #norms = np.linalg.norm(datasn[:,[2,3]], axis=1, keepdims=True)
    #normalized_unit_length = points / norms
   # datasn[:,[2,3]]=datasn[:,[2,3]]/norms23
   # datasn[:,[5,6]]=datasn[:,[5,6]]/norms56
    return datasn
#datasn=normalize_dataset(datasn)
#datasnplot = reshapeperidped(datasn)

datas=np.array(datas)
#datafit = datas[:, [5,6]]
datafit = datas[:, [4]]
datafit=datafit.reshape(-1,1)
bandwidth = estimate_bandwidth(datafit, quantile=0.2, n_samples=100)
#clustersms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
n_initial_cluster=8
clustersK = AgglomerativeClustering(n_clusters=n_initial_cluster, linkage='complete', metric='manhattan')#create first cluster with 20 maximum direction cluster
#clustersK = KMeans(n_clusters=20,init='k-means++',max_iter=250)
#resK = clusters.fit(datafit)
res= clustersK.fit(datafit)
#resK= clustersms.fit(datafit)
#plt.gca().invert_yaxis()

xpoints = [sublist[2] for sublist in datas]
ypoints = [sublist[3] for sublist in datas]
#vx = datas[:, 5]
#vy = datas[:, 6]
pedid=[sublist[1] for sublist in datas]

def arrangedatapercluster(idclusters, lbls, datafit,dataupdate):
    #Sort the data by cluster; if an idcluster is found in lbls, then it is added to arranged.
    arranged = []
    i=0
    for l in range(0, len(lbls)):
        if (idclusters == lbls[l]):
            val = datafit[l]
            val[-1] = idclusters
            dataupdate[l][-1] = idclusters
            datafit[l][-1]=idclusters
            val2=np.append(val,i)
            arranged.append(val2)#-1 is the cluster index
        i+=1
    return arranged

def arrangeddatas(maxclusters, lbls, datafit, dataupdate):
    #maximum cluster, labels, arranged data, main data that the cluster label will be updated
    result=[]
    for idcluster in range(0, maxclusters):
        result.append([idcluster, arrangedatapercluster(idcluster, lbls, datafit,dataupdate)])
    return result

#def createSubcluster(labels, datafit,clustertype):
maxclus=max(res.labels_)+1
result = arrangeddatas(maxclus, res.labels_, datas,datas)
u = 0
subclusters=[]
for sub in result:
    #subclusters = []
    sresk=None
    if(len(sub[1])>1):
        sclustersK = AgglomerativeClustering(distance_threshold=tdist,n_clusters=None, linkage='complete', metric='manhattan')
        sdata=np.array(sub[1])
        sresK = sclustersK.fit(sdata[:, [2, 3]])
    else:
        sresK=None
    subclusters.append([sub[0],sresK])

colors = mcolors.CSS4_COLORS
colordict=[]
for co in colors:
    colordict.append(co)
#colordict = ['red', 'green', 'black', 'yellow', 'purple', 'brown', 'cyan', 'magenta', 'blue', 'orange', 'olive']
#markerdict=['o','x','+','1','2','3','4','8','s','P','p','*','H','D','X','V','^']
markerdict=['o','x','+','1','2','3','4','s','P','*']
labels = []
labelsMarker=[]
for a in res.labels_:
    labels.append(colordict[int(a)])
# for a in res.labels_:
#     labelsMarker.append(markerdict[int(a)])
labelsK = []
slabelsK = []
lbl = 0
#input the subcluster label to the main data
def idperfirstcluster(subresult):
    ids=[0]
    initid=0
    for idx,h in enumerate(subresult):
        if(idx<len(subresult)-1):
            if(h[1]!=None):
                ids.append(h[1].labels_.max()+initid+1)
                initid=h[1].labels_.max()+initid+1
            else:
                 initid=initid+1
                 ids.append(initid)
    return ids
idstart=idperfirstcluster(subclusters)
for d in datas:
    cari=[]
    for h in range(0, len(result)+1):#result is data arrange cluster
        hasilke = np.array(result[h][1])[:,1]#grab id data
        cari = np.where(d[1] == hasilke)#find the id value on d 
        if (len(cari[0]) > 0):#if found then break
            break
    try:    
        subke = subclusters[h]
        if (subke[1] == None):
            slabelsK.append([subke[0],-1])    
        else:
            find=cari[0][0]
            ids=subke[1].labels_[find]+idstart[h]
            slabelsK.append([subke[0],ids])
        #find label per pedestrian id
    except Exception as E:
        s=subke
        print('no 1 ',E)
slabelsK=np.array(slabelsK)
uniqlabel=slabelsK[np.unique(slabelsK[:,[0, 1]], axis=0, return_index=True)[1]]
idx=0
for idx,slbl in enumerate(slabelsK):
    if(idx==130):
        a=0
    index=np.where((slbl[0]==uniqlabel[:,0]) & (slbl[1]==uniqlabel[:,1]))
    labelsK.append([index[0][0],[slbl]])

lbl=0

EvKmeans = []
EvHierar = []
EvMS = []
EvCKmeans = []
EvCHierar = []
EvCMS = []

def idxreturnpedestrian(anno,personID):
    #find posisition based on annotation
    return np.where(personID == anno)
def idxreturncluster(anno,clusterID):
    #find cluster posisition based on annotation
    result=np.where(anno == np.array(clusterID))
    return result[0]
evflag=[]
def evpercluster(member, isdist,threshold,frame,clusterno):
    #return true if there are outilers, return false when no outlier
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor
    from scipy.spatial import distance_matrix, distance
    isdist=False
   
    if(isdist):
        dist = member[:, (2,3)]
    else:
        dist =member[:, (4)].reshape(-1,1)
      #
    #menggunakan teknik LOF
   # a=len(dist)
    # if(len(dist)==0):
    #     rest=[0]
    #     status=False
    if(len(member)>2):
        status=False
        #if(member[0][-2]==22 and (frame==240 or frame==300)):#ngecek no 92 apakah kena threshold
       #     print('tes)
        #    evflag.append('tes')
            #beberapa data ga di kick dari cluster
        if (np.isnan(dist).any() == False):  

            n=int(math.floor(len(member)*(80/100)))
            
            contaminations = 0.2
            clf = LocalOutlierFactor(n_neighbors=n, p=2,contamination=contaminations)
            
            result = clf.fit_predict(dist)
            resultneg = clf.negative_outlier_factor_
                
            rest = np.where(result == -1)#find all anomalies
            if (len(rest[0]) > 0):
                status=True
            if(status!=True):    
                dist = member[:, (2,3)]
                clf = LocalOutlierFactor(n_neighbors=2, p=2,contamination=contaminations)
                result = clf.fit_predict(dist)
                rest = np.where(result == -1)
                if (len(rest[0]) > 0):
                  status=True
        else:
            rest = [[1]]
    elif (len(member)==2):#If the number of members is two, then directly check: if the distance (Euclidean) is greater than the threshold, it will be considered an anomaly.
        dist =member[:, (4)].reshape(-1,1)
        dist2= member[:, (2,3)]
        dista = smallest_angular_distance(dist[0],dist[1])
        dista2=distance.euclidean(dist2[0],dist2[1])

        if (dista > threshold):
            rest = [[1]]
            status = True
        elif(dista2>tdist):
            rest=[[1]]
            status=True
        else:
            rest = []
            status=False
    
    return rest,status 

def nowhasmember(idcentro,datapeds):
    state=False

    for ped in datapeds:
        if(idcentro==ped[-1]):
            state=True
            break
    return state
def findnearestradius(ped, datanormal,threshold,centro):
    #find neighbourhood that is in radius
    from scipy.spatial import distance_matrix, distance
    neighbors = []
    distances = []
    directions=[]
    clusternear = []
    idx = []
    accumulativedistance=[]
    if(len(centro)>0):
        try:
            centronormal = centro.copy()
            centronormal=normalize_dataset(np.array(centronormal))
            
            i = 0

            for datan in centronormal:
                pointA = [ped[2],ped[3]]
                pointB = [datan[2], datan[3]]
                pointAdir =ped[4]
                pointBdir = datan[4]
                distadir=smallest_angular_distance(pointAdir,pointBdir)
                dista = distance.euclidean(pointA, pointB)
                #find if the distance under threshold
                
                if(distadir<threshold[0]):
                    if (dista < threshold[1]):
                        if(datan[1]!=ped[1] and not(math.isnan(datan[-1]))):
                          #if(nowhasmember(datan[1],datanormal)==True):
                            neighbors.append(datan)
                            distances.append(dista)
                            directions.append(distadir)
                            if(distadir>100):
                                a=0
                            idx.append(i)
                i += 1

            distances = np.array(distances)
            directions=np.array(directions)
            accumulativedistance = np.array([])
        except Exception as e:
            print('no 2 ',e)
        try:
            if (len(neighbors) > 0):
                #if (len(distances) == 1):
                #    print('test')
                normaldistances = distances#preprocessing.normalize([distances])
                normaldirections=directions##preprocessing.normalize([directions])
                #accumulativedistance=np.add(normaldistances,normaldirections)#melakukan penjumlahan distance direction dalam kondisi normal
                accumulativedistance=normaldirections
                
                clusternear = np.array(neighbors)[:, 1]  #mengindikasikan bahwa nomor cluster
        except Exception as e:
            print('no 3 ',e)
        #return neighbour data, accumulative distance, closest cluster id, and neighbour id
    return neighbors,accumulativedistance,clusternear,idx
additionalchange = []
            
def evaluatecluster(label, datanormal, dataclust, datasupdate, frame, centro,maxcluster):
    theta=[tdirect,tdist]#direction,distance
    b=datasupdate
    ix = []
    for percluster in dataclust: #looping percluster
        if (len(percluster[1]) > 1):#what if less than 2
            member = np.array(percluster[1])
            if(percluster[0]==1):
                a=0
            if(frame==800):
                if(percluster[0]==326):
                    a=1
            disevres,disstat = evpercluster(member, True,theta[0],frame,percluster[0]) 
            
            #direvres, dirstat = ev(member, False,0.5)
            if (disstat == True):#jika ada yang diubah
                for idex in disevres:
                    #tunjukan perubahaannya
                    change = member[idex]
                    if (len(idex) == 1):
                        try:
                            #change = member[idex]
                            neighbors, dist, clustnear, indexofresult = findnearestradius(change[0], datanormal, theta,centro)#cari cluster 
                            if(len(dist)>0): #jika ditemukan
                                val = neighbors[np.argmin(np.array(dist))][-1]
                                indexneighbors=int(indexofresult[np.argmin(np.array(dist))])
                                vallabel= centro[indexneighbors]
                                if(vallabel[1]==60 and frame ==245):
                                    a=0

                                datasupdate[int(change[0][-1])][-1] = vallabel[1]#int(-1) 
                                label[int(change[0][-1])] = [vallabel[1],-1]
                                pointA = datasupdate[int(change[0][-1])][4]#,datasupdate[int(change[0][-1])][6]]
                                pointB = vallabel[4]
                                cenor=centro.copy()
                                cenor=normalize_dataset(np.array(cenor))
                                val22=cenor[indexneighbors]
                                pointA1=datanormal[int(change[0][-1])][4]#,datanormal[int(change[0][-1])][6]]
                                poinB1=val22[4]#,val22[6]]
                                dista2=smallest_angular_distance(pointA1,poinB1)
                                dista=smallest_angular_distance(pointA,pointB)
                                

                                additionalchange.append(datasupdate[int(change[0][-1])])
                            else:
                                #indexneighbors=int(indexofresult[np.argmin(np.array(dist))])
                                vallabel= -1#label[indexneighbors]
                                datasupdate[int(change[0][-1])][-1] = vallabel
                                label[int(change[0][-1])] = [vallabel,vallabel]

                                additionalchange.append(datasupdate[int(change[0][-1])])
                            #label[int(change[0][-1])][0]=11#konek nih sayyyyyy
                        except Exception as e:
                                print('no 4 ',e)
                    else:
                        for c in change:
                            try:
                                
                                neighbors, dist, clustnear,indexofresult = findnearestradius(c, datanormal, theta,centro)

                                if(len(dist)>0):
                                    idxdis=np.argmin(np.array(dist))
                                    val = neighbors[np.argmin(np.array(dist))][-1]
                                    indexneighbors=int(indexofresult[np.argmin(np.array(dist))])
                                    #vallabel= label[indexneighbors]
                                    vallabel= centro[indexneighbors]
                                    datasupdate[int(c[-1])][-1] = vallabel[1]#int(-1)
                                    label[int(c[-1])] = [vallabel[1],-1]
                                   # print('masuk 3')
                                    pointA = datasupdate[int(c[-1])][4]#,datasupdate[int(c[-1])][6]]
                                    pointB = vallabel[4]#, vallabel[6]]
                                    cenor=centro.copy()
                                    cenor=normalize_dataset(np.array(cenor))
                                    val22=cenor[indexneighbors]
                                    pointA1=datanormal[int(c[-1])][4]#,datanormal[int(c[-1])][6]]
                                    poinB1=val22[4]#,val22[6]]
                                    dista2=smallest_angular_distance(pointA1,poinB1)
                                    dista=smallest_angular_distance(pointA,pointB)
                                    #print(int(c[-1]))
                                    additionalchange.append(datasupdate[int(c[-1])])
                                else:
                                #indexneighbors=int(indexofresult[np.argmin(np.array(dist))])
                                    vallabel= -1#label[indexneighbors]
                                    datasupdate[int(c[-1])][-1] = vallabel
                                    label[int(c[-1])] = [vallabel,vallabel]
                                   # print('frame:',frame)
                                   # print('masuk 4 dihilangkan')
                                    #print(int(c[-1]))
                                    additionalchange.append(datasupdate[int(c[-1])])
                            #label[int(change[0][-1])][0]=11#konek nih sayyyyyy
                                #label[int(change[0][-1])][0]=11
                            except Exception as e:
                                print('no 5 ',e)
                    b = datasupdate[idex]

    for idx,data in enumerate(datasupdate): #find new data
        clustername = data[7]
        if (math.isnan(data[7]) or data[7]==-1):#jika merupakan data baru
                neighbors, dist, clustnear, indexofresult = findnearestradius(data, datanormal, theta,centro)#cari yang paling dekat dengan c entroid       
                if (len(neighbors) == 0):#if not find put to the storage
                    nancluster.append(data)
                else:# if found then find the nearest distance
                    try:
                        idxmin=np.argmin(dist)
    
                        label[idx] = [clustnear[idxmin], -1]
                        datasupdate[idx][-1] = clustnear[idxmin]
                        additionalchange.append(datasupdate[idx])
                        a = 0
                    except Exception as e:
                        print('no 6 ',e)

    if(len(nancluster)>5):#If a cluster that is not found has a nearest radius greater than 5, or if it does not belong to any cluster (cluster value is -1), then
        nanclusters = np.array(nancluster)
        predictedval=nanclusters[:,4].reshape(-1,1)
     #   arr=[[predictedval[0][0],predictedval[0][1]]]
     #   testpredic=res.predict(arr)
        agglo2 = AgglomerativeClustering(distance_threshold=theta[0],n_clusters=None)#melakukan cluster ulang dengan jarak degree maksimal
        #rescluster = agglo.fit(predictedval)
        resultnewclust = agglo2.fit_predict(predictedval)
        if(len(evflag)>0):
            a=0
 
        resultnewclustunique = np.unique(resultnewclust)
        nangroup = []
        for nan in resultnewclustunique:#Grouping the new data based on the new clusters.
            nansubgroup=[]
            for idxes,nancluster in enumerate(nanclusters):
                if (resultnewclust[idxes]==nan):
                    nansubgroup.append(nancluster)
            nangroup.append(nansubgroup)
        #subclusternew=[]
        lastclusternumber = maxcluster
        for nanfit in nangroup:#Perform clustering on the clusters: -1 indicates no subcluster, 0 for the first cluster, and 1 for the second cluster.
            if(len(nanfit)>1):
               # print('masuk9')
                agglo = AgglomerativeClustering(distance_threshold=theta[1],n_clusters=None)
                valls=np.array(nanfit)[:, [2, 3]]
                rescluster = agglo.fit(np.array(nanfit)[:, [2, 3]])
                labelsrescluster=rescluster.labels_.copy()
                #lastclusternumber = maxcluster
                clu1=[]
                clu2=[]
                for idx,lbl in enumerate(labelsrescluster):
                    if(lastclusternumber==182):
                        axx=0
                    if(idx in labelsrescluster):
                        lastclusternumber+=1
                        labelsrescluster[labelsrescluster == idx] = lastclusternumber
                #labelsrescluster[labelsrescluster == 1] = lastclusternumber + 2
                    
                a = 0
                for i, lc in enumerate(labelsrescluster):
                    dataid = datasupdate[:, 1]
                    idtosearch=nanfit[i][1]
                    index = np.where(dataid==idtosearch)[0][0]
                    datasupdate[index][-1] = lc
                    if(lc==183):
                        sada=0
                    label[index] = [lc, -a]
                    additionalchange.append(datasupdate[index])

            else:

                rescluster = lastclusternumber + 1
                if(rescluster==183):
                    aa=1
                lastclusternumber+=1
                dataid = datasupdate[:, 1]
                idtosearch=nanfit[0][1]
                index = np.where(dataid==idtosearch)[0][0]
                cek=np.where(rescluster==np.array(datasupdate)[:,-1])[0]
                if(len(cek)>0):
                    a=0                
                datasupdate[index][-1] = rescluster
                label[index] = [rescluster, -900]
                additionalchange.append(datasupdate[index])
                if(rescluster==60 and frame ==245):
                    a=0

    return label
wrong = []
def calculatedistanceinacluster(clu1):#Calculate the distance within a cluster to its mean value
    avgdisx=avgdisy=0
    for c1 in clu1:
        avgdisx+=c1[4]
   
    avgdisx=avgdisx/len(clu1)
    distan1=0
    for c1 in clu1:
        distan1+=smallest_angular_distance(avgdisx,c1[4])
    distan1=distan1/len(clu1)
    return distan1
def assignClusterID(searchpersonIDs, personIDcluster, clusterlabels):
    #arrange the clusster based on input order
    cluster=[]
    for pID in searchpersonIDs:
        try:
            find = personIDcluster.index(pID)
            cluster.append(clusterlabels[find])
        except ValueError:
            cluster.append([-1,-1])
    return cluster
def changelabelSK(newdata, olddata, oldlabel,newlabel):
    try:
        i = 0
        a=0
        for db in newdata:
            find = -1
         
            try:
                find = olddata.index(db)
                #if (labellama[find][0] == 25):
               #     a+=1
            except Exception as e:
                find=-1
            if (find >= 0):
                oldlabel[find] = newlabel[i]
            i += 1
        a=0
    except Exception as e:
        print('no 7 ',e)
def updatenewcluster(labelsc, datasupdate, additionalchange):
    for idx,du in enumerate(datasupdate):
        for ac in additionalchange:
            if (du[1] == ac[1]):
                du[-1] = ac[-1]#change the last value
                labelsc[idx]=[ac[-1],-1]
# def findrealradius(clusperframe,xcentro,ycentro):
#     for clus in clusperframe:
#         eu=euclidean(clusper)    
def findcentroid(datapercluster,prevdatapercluster,frame,begining,prevcentroids,allprevcentroids):
#mfind centroid based on cluster
    centroids = []

    temp=[] 
    prevdataperclusternotnull=[]
    for prevc in prevdatapercluster:
        if(len(prevc[1])>0):
            prevdataperclusternotnull.append(prevc)
    for clus2 in datapercluster:
        if(frame==begining):
            clus = np.array(clus2[1])
            n = len(clus)
            try:
                if(n>1):
                    dist=[]
                    x = sum(clus[:, 2])/n
                    y = sum(clus[:, 3])/n
                    V=sum(clus[:,4])/n
                    Vx = sum(clus[:, 5])/n
                    Vy=sum(clus[:,6])/n
                    points=clus[:, (2,3)]
                    distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                    d=distances.max()
                    if(d>450):
                        aas=0
                    #r=findrealradius(clus,x,y)
                    if(V<0):
                        a=0
                    centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d,n])  #frame,idcluster,x,y,V,Vx,Vy,radius,jumlah cluster,
                else:
                    centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
            except Exception as E:
                a=clus
        else:#if it's not the begining
            clus = np.array(clus2[1])
           
            n = len(clus)
            if(n>0):#if the cluster has member
                if(clus[0][-2]==183):
                    sdasd=0
                listofcentro=[i[0] for i in prevdatapercluster]
                previdx=np.where(clus[0][-2]==np.array(listofcentro))[0]
                if(len(previdx)>0):#if there is a previous cluster
                    prevlist=np.array(prevdatapercluster[previdx[0]][1])#get the previous cluster
                    if(len(prevlist)>0):#if the member found
                        num=0
                        deltax=deltay=deltav=deltavx=deltavy=0
                        for member in clus:
                            idxpeds=np.where(member[1]==prevlist[:,1])[0]#find pedestrian id that equal with prev list
                            if(len(idxpeds)>0):
                                num+=1
                                deltax+=member[2]-prevlist[idxpeds[0]][2]
                                deltay+=member[3]-prevlist[idxpeds[0]][3]
                
                        if(num>0):#jika isinya ada
                            indexprevclus=np.where(clus[0][-2]==np.array(prevcentroids)[:,1])[0]
                            if(len(indexprevclus)>0):#If the centroid data is found in the cluster from the previous frame.
                                x=prevcentroids[indexprevclus[0]][2]+(deltax/num)
                                y=prevcentroids[indexprevclus[0]][3]+(deltay/num)
                               # if(clus[0][-2]==34 and frame%10==0):
                               #     print('stop')
                                #Vx=(prevcentroids[indexprevclus[0]][5]+(prevcentroids[indexprevclus[0]][2]-x))/2
                                #Vy=(prevcentroids[indexprevclus[0]][6]+(prevcentroids[indexprevclus[0]][3]-y))/2
                                Vx=(prevcentroids[indexprevclus[0]][2]-x)
                                Vy=(prevcentroids[indexprevclus[0]][3]-y)
                                V=calculate_direction(Vx,Vy)
                                points=clus[:, (2,3)]
                                distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                                d=distances.max()
                                # if(d>0 and n==1 and clus[0][-2]==0):
                                #     z=0
                                centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d,n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
                            else:
                                a=0
                        else:
                            indexprevclus=np.where(clus[0][-2]==np.array(prevcentroids)[:,1])[0]
                            if(len(indexprevclus)>0):
                                centroids.append([clus[0][0], clus[0][-2], prevcentroids[indexprevclus[0]][2], prevcentroids[indexprevclus[0]][3], prevcentroids[indexprevclus[0]][4], prevcentroids[indexprevclus[0]][5], prevcentroids[indexprevclus[0]][6],prevcentroids[indexprevclus[0]][7],n])
                    else:#if previous member does not exist
                        if(n>1):#if the member more than 1, reidentification appear then it will jump
                            x = sum(clus[:, 2])/n
                            y = sum(clus[:, 3])/n
                            V=sum(clus[:,4])/n
                            Vx = sum(clus[:, 5])/n
                            Vy=sum(clus[:,6])/n
                            points=clus[:, (2,3)]
                            distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                            d=distances.max()
                            if(V<0):
                                a=0
                            centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d,n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
                        else:#if the member 1
                            #centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],n])
                            indexprevclus=np.where(clus[0][-2]==np.array(prevcentroids)[:,1])[0]
                            if(len(indexprevclus)>0):#kalau centroid sebelumnya ada
                                centroids.append([clus[0][0], clus[0][-2], prevcentroids[indexprevclus[0]][2], prevcentroids[indexprevclus[0]][3], prevcentroids[indexprevclus[0]][4], prevcentroids[indexprevclus[0]][5], prevcentroids[indexprevclus[0]][6],prevcentroids[indexprevclus[0]][7],n])
                            else:#kalau centroid ga ada diframe sebelumnya, semua masalah ada disni cuy, ganti pakai all previous centroid
                                tempo=0
                                #centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],n])
                                for preframe in reversed(allprevcentroids):
                                    for precent in preframe:
                                        if(precent[1]==clus[0][-2]):
                                            tempo=1
                                            centroids.append([clus[0][0], precent[1], precent[2], precent[3],precent[4],precent[5],precent[6],precent[7],n])#this possible jump, cluster gone then pop up again.
                                            break
                                    if(tempo==1):
                                        break
                                if(tempo==0):
                                    centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                                    
                                            
                else:# if no previous cluster
                    if(n>1):#if the member more than one
                        x = sum(clus[:, 2])/n
                        y = sum(clus[:, 3])/n
                        V=sum(clus[:,4])/n
                        Vx = sum(clus[:, 5])/n
                        Vy=sum(clus[:,6])/n
                        points=clus[:, (2,3)]
                        distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                        d=distances.max()
                        centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d, n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
                    else:#if the member is only one
                        centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                        #centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],n])
            else:
                if(len(clus)==1):
                    centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                else:
                   # temp.append(clus2[0])
                    #print(temp)
                    I=0
                


    return centroids
def findcentroidtraining(datapercluster,prevdatapercluster,frame,begining,prevcentroids,allprevcentroids):
#find centroid based on percluster data
    centroids = []

    if(frame==81):
        i=0
    temp=[] 
    prevdataperclusternotnull=[]
    for prevc in prevdatapercluster:
        if(len(prevc[1])>0):
            prevdataperclusternotnull.append(prevc)
    for clus2 in datapercluster:
        if(frame==begining):
            clus = np.array(clus2[1])
            n = len(clus)
            try:
                if(n>1):
                    dist=[]
                    x = sum(clus[:, 2])/n
                    y = sum(clus[:, 3])/n
                    V=sum(clus[:,4])/n
                    Vx = sum(clus[:, 5])/n
                    Vy=sum(clus[:,6])/n
                    points=clus[:, (2,3)]
                    distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                    d=distances.max()
                    #r=findrealradius(clus,x,y)
                    if(V<0):
                        a=0
                    centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d,n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster,tambahkan radius
                else:
                    centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
            except Exception as E:
                a=clus
        else:#jika bukan awal
            clus = np.array(clus2[1])
            
            n = len(clus)
            if(n>0):#jika cluster ada anggotanya
                listofcentro=[i[0] for i in prevdatapercluster]
                previdx=np.where(clus[0][-2]==np.array(listofcentro))[0]#find cluster index in previous frame
                if(len(previdx)>0):#if the cluster exist
                    prevlist=np.array(prevdatapercluster[previdx[0]][1])#find previous cluster
                    if(len(prevlist)>0):#if the member exist
                        num=0
                        deltax=deltay=deltav=deltavx=deltavy=0
                        for member in clus:
                            idxpeds=np.where(member[1]==prevlist[:,1])[0]#find pedestrian id that equal with previous list
                            if(len(idxpeds)>0):
                                num+=1
                                deltax+=member[2]-prevlist[idxpeds[0]][2]
                                deltay+=member[3]-prevlist[idxpeds[0]][3]
                    
                        if(num>0):
                            indexprevclus=np.where(clus[0][-2]==np.array(prevcentroids)[:,1])[0]
                            if(len(indexprevclus)>0):#if the centroiud is found from previous frame
                                x=prevcentroids[indexprevclus[0]][2]+(deltax/num)
                                y=prevcentroids[indexprevclus[0]][3]+(deltay/num)
                              #  if(clus[0][-2]==34 and frame%10==0):
                              #      print('stop')
                                #Vx=(prevcentroids[indexprevclus[0]][5]+(prevcentroids[indexprevclus[0]][2]-x))/2
                                #Vy=(prevcentroids[indexprevclus[0]][6]+(prevcentroids[indexprevclus[0]][3]-y))/2
                                Vx=(prevcentroids[indexprevclus[0]][2]-x)
                                Vy=(prevcentroids[indexprevclus[0]][3]-y)
                                V=calculate_direction(Vx,Vy)
                                points=clus[:, (2,3)]
                                distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                                d=distances.max()
                                # if(d>0 and n==1 and clus[0][-2]==0):
                                #     z=0
                                centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d,n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
                            else:
                                a=0
                        else:
                            indexprevclus=np.where(clus[0][-2]==np.array(prevcentroids)[:,1])[0]
                            if(len(indexprevclus)>0):
                                 centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                                #centroids.append([clus[0][0], clus[0][-2], prevcentroids[indexprevclus[0]][2], prevcentroids[indexprevclus[0]][3], prevcentroids[indexprevclus[0]][4], prevcentroids[indexprevclus[0]][5], prevcentroids[indexprevclus[0]][6],prevcentroids[indexprevclus[0]][7],n])
                    else:#if the list of previous data is found
                        if(n>1):#kalau anggotanya lebih dari 1, reidentification appear then it will jump
                            x = sum(clus[:, 2])/n
                            y = sum(clus[:, 3])/n
                            V=sum(clus[:,4])/n
                            Vx = sum(clus[:, 5])/n
                            Vy=sum(clus[:,6])/n
                            points=clus[:, (2,3)]
                            distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                            d=distances.max()
                            if(V<0):
                                a=0
                            centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d,n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
                        else:#if the member is only one
                            #centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],n])
                            indexprevclus=np.where(clus[0][-2]==np.array(prevcentroids)[:,1])[0]
                            if(len(indexprevclus)>0):#kalau centroid sebelumnya ada
                                centroids.append([clus[0][0], clus[0][-2], prevcentroids[indexprevclus[0]][2], prevcentroids[indexprevclus[0]][3], prevcentroids[indexprevclus[0]][4], prevcentroids[indexprevclus[0]][5], prevcentroids[indexprevclus[0]][6],prevcentroids[indexprevclus[0]][7],n])
                            else:#if the centroid doesn not exist in previus frame
                                tempo=0
                  
                                centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                                            
                else:# if no cluster member
                    if(n>1):#if the member more than one
                        x = sum(clus[:, 2])/n
                        y = sum(clus[:, 3])/n
                        V=sum(clus[:,4])/n
                        Vx = sum(clus[:, 5])/n
                        Vy=sum(clus[:,6])/n
                        points=clus[:, (2,3)]
                        distances = np.array([distance.euclidean(p, [x,y]) for p in points])
                        d=distances.max()
                        centroids.append([clus[0][0], clus[0][-2], x, y, V, Vx, Vy,d, n])  #frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
                    else:#if the member only one
                        centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                        #centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],n])
            else:
                if(len(clus)==1):
                    centroids.append([clus[0][0], clus[0][-2], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6],0,n])
                else:
                   # temp.append(clus2[0])
                    #print(temp)
                    I=0
                

    return centroids

def getdataupdatesbyframe(datasupdate,frame):
    res=[]
    for data in datasupdate:
        if(data[0][0]==frame):
            res=data
            break
    return res
def calculateDistanceInside(centroid, clustermembers,labels,datasupdates):
    res = []
    centroidpoint = np.array(centroid)[:, 4]
    indexcentro=0
    try:
        for idx, (cid,cm) in enumerate(clustermembers):#ga semua member punya anggota
            if(len(cm)>0):
                points = np.array(cm)[:, 4]
                dist=0
                d=[]
                if(cid==centroid[indexcentro][1]):
                    for point in points:
                        d.append(smallest_angular_distance(centroidpoint[indexcentro],point))
                        dist += np.sum(smallest_angular_distance(centroidpoint[indexcentro],point))  #menggunakan Average Score untuk hitung error distancenya
                    dist = dist/len(points)#count if more than 1 pedestrian, but the problem is when the cluster is only consist of 1 member
                else:
                    print(cid)

                res.append([cid, dist,len(points)])
                indexcentro+=1
    except Exception as e:
        a=0
    return res
def add_1d_arrays(arr1, arr2):
    # Convert input lists to numpy arrays
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    
    # Determine the maximum length
    max_len = max(len(arr1), len(arr2))
    
    # Extend the arrays with zeros to match the maximum length
    arr1_padded = np.pad(arr1, (0, max_len - len(arr1)), 'constant')
    arr2_padded = np.pad(arr2, (0, max_len - len(arr2)), 'constant')
    
    # Add the two arrays element-wise
    result = np.concatenate(arr1_padded,arr2_padded)
    return result

def evaluationpercluster(centroid, distance_evaluation):
    datafull=[]#np.array(centroid[0])[:,1]
    for idx,arrcentro in enumerate(centroid):
        for ac in arrcentro:
            datafull.append(ac[1])
    ClusterUnique = np.unique(np.array(datafull))
    evalresult = []
    
    for CU in ClusterUnique:
        evalpercluster=[]
        for frame, arrayevaluation in distance_evaluation: 
            #if(frame%5==0 or frame==75): 
            for ae in arrayevaluation:  
                if(CU==ae[0]): #if the cluster value is the same
                    evalpercluster.append([frame, ae])
                    break
        evalresult.append([CU,evalpercluster])
    return evalresult  
def smoothnessevaluationpercluster(centroid,threshold):#Used to evaluate changes in distance for pedestrians and convert the distance format to per-cluster.
    evsmooth=[]
    #errtotal=[]
    for idx,arrcentroframe in enumerate(centroid):
        errperframe=[]
        for ac in arrcentroframe:
            if(idx==0):
                val=[ac[0],0,0,0,ac[2],ac[3],ac[4],ac[5],ac[6],0]
                val3=[]
                val3.append(val)
                val2=[ac[1],val3]
                evsmooth.append(val2)#cluster id, distance deviation, degree direction, direction deviation,x,y,d,vx,vy
            else:
               # if(idx==1):
               #     evsmooth=np.array(evsmooth)
                clustername=[i[0] for i in evsmooth]
                countcluster=np.where(ac[1]==clustername)
                if(len(countcluster[0])>0):
                    
                    deltadistance=math.sqrt((math.pow(ac[2]-evsmooth[countcluster[0][0]][1][-1][4],2)+math.pow(ac[3]-evsmooth[countcluster[0][0]][1][-1][5],2)))
                    deltadegree=math.sqrt((math.pow(ac[4]-evsmooth[countcluster[0][0]][1][-1][6],2)))
                    deltadirection=math.sqrt((math.pow(ac[5]-evsmooth[countcluster[0][0]][1][-1][7],2)+math.pow(ac[6]-evsmooth[countcluster[0][0]][1][-1][8],2)))
                  

                    #evsmooth[1][countcluster[0][0]][1]=evsmooth[1][countcluster[0][0]][1]+deltadistance
                    #evsmooth[1][countcluster[0][0]][2]=evsmooth[1][countcluster[0][0]][2]+deltadirection
                    valdist=evsmooth[countcluster[0][0]][1][-1][1]+deltadistance
                    valdir=evsmooth[countcluster[0][0]][1][-1][3]+deltadirection
                    valdeg=evsmooth[countcluster[0][0]][1][-1][2]+deltadegree
                    evsmooth[countcluster[0][0]][1].append([ac[0],valdist,valdeg,valdir,ac[2],ac[3],ac[4],ac[5],ac[6],deltadistance])
                else:
                    val=[ac[0],0,0,0,ac[2],ac[3],ac[4],ac[5],ac[6],0]
                    val3=[]
                    val3.append(val)
                    val2=[ac[1],val3]
                    evsmooth.append(val2)
            frame=ac[0]

        #evsmooths.append([frame,evsmooth.copy()])
    return evsmooth
def smoothnessevaluationperpedestrian(df,begin,end):
    evsmoothped=[]
    for number in range(begin, end):    
        alldataonframe = getdatabyframe(df, number)
        for ap in alldataonframe:
            if(number==begin):
                val=[ap[0],0,0,0,ap[2],ap[3],ap[4],ap[5],ap[6]]
                val3=[]
                val3.append(val)
                val2=[ap[1],val3]
                evsmoothped.append(val2)
            else:
                clustername=[i[0] for i in evsmoothped]
                countped=np.where(ap[1]==clustername)
                if(len(countped[0])>0):
                        # a=evsmooth[1]
                        # b=countcluster[0][0]
                        # c=evsmooth[1][countcluster[0][0]][3]
                        
                    deltadistance=math.sqrt((math.pow(ap[2]-evsmoothped[countped[0][0]][1][-1][4],2)+math.pow(ap[3]-evsmoothped[countped[0][0]][1][-1][5],2)))
                    deltadirection=math.sqrt((math.pow(ap[5]-evsmoothped[countped[0][0]][1][-1][7],2)+math.pow(ap[6]-evsmoothped[countped[0][0]][1][-1][8],2)))
                    deltadegree=math.sqrt((math.pow(ap[4]-evsmoothped[countped[0][0]][1][-1][6],2)))
                        #evsmooth[1][countcluster[0][0]][1]=evsmooth[1][countcluster[0][0]][1]+deltadistance
                        #evsmooth[1][countcluster[0][0]][2]=evsmooth[1][countcluster[0][0]][2]+deltadirection
                    valdist=evsmoothped[countped[0][0]][1][-1][1]+deltadistance
                    valdir=evsmoothped[countped[0][0]][1][-1][3]+deltadirection
                    valdegree=evsmoothped[countped[0][0]][1][-1][2]+deltadegree
                    evsmoothped[countped[0][0]][1].append([ap[0],valdist,valdegree,valdir,ap[2],ap[3],ap[4],ap[5],ap[6]])
                else:
                    val=[ap[0],0,0,0,ap[2],ap[3],ap[4],ap[5],ap[6]]
                    val3=[]
                    val3.append(val)
                    val2=[ap[1],val3]
                    evsmoothped.append(val2) 
    return evsmoothped
        #npedesnframe.append([alldataonframe[0][0],len(alldataonframe)])
def sumclustermembereveryframes(centroids):
    #get total number of pedestrian every frames from every cluster
    frameandtotal=[]
    for centroframe in centroids:
        totalpeds=0
        for centroid in centroframe:
            totalpeds += centroid[-1]
        frameandtotal.append([centroframe[0][0], totalpeds])
    return frameandtotal
def countpedestrianperframes(df, begin, end):
    #get total number of pedestrian every frames from GT
    npedesnframe=[]
    for number in range(begin, end):    
        alldataonframe = getdatabyframe(df, number)
        npedesnframe.append([alldataonframe[0][0],len(alldataonframe)])

    return npedesnframe
def getpedistriandatafromcluster(clusterinput):
    clustersearch=[]
    
    for cluin in clusterinput:
        percluster=[]
        perpeds=[]
        for idx,mf in enumerate(membercentro):
            if(idx==85):
                a=0
            for clust in mf:
                if(cluin==clust[0]):
                    for peds in clust[1]:
                        perpeds.append(peds)
        clustersearch.append(perpeds)
    finalclust=[]
    for clus in clustersearch:
        #uniq=np.unique(np.array(clus)[:,1])
        groups = defaultdict(list)
        for peds in clus:
            groups[int(peds[1])].append(peds)
        grouped_dict = dict(groups)
    # Convert lists to numpy arrays (optional, if you need arrays as values)
        grouped_arrays = {k: np.array(v) for k, v in grouped_dict.items()}
        finalclust.append(grouped_arrays)
            #peds=np.array(peds)#array of pedestrian every frame
            #axdetails.plot(peds[:,2],peds[:,3])
    return finalclust
def findmaxcluster(centroids):
    maxx=0
    for framec in centroids:
        for centr in framec:
            if(maxx<centr[1]):
                maxx=centr[1]
    return maxx
def findmaxclusterbegin(datasupdate):
    maxx=0
    for data in datasupdate:
        if(~math.isnan(data[-1])):
            if(maxx<data[-1]):
                maxx=data[-1]
    return maxx
def runtheframe(begining,ending):
    maxframe = ending#int(max(np.array(df)[:, 0]))
    centroids = []
    disteval = []
    centro=[]
    dataclusterprevious=[]
    prevcentroids=[]
    membercentroid=[]
    datasupdates=[]
    for frame in range(begining,maxframe):
        dataA = getdatabyframe(df,frame)
        labelsMark = []
        labelscolors1=[]
        datasupdate = np.array(dataA) 
        #datas=normalize_dataset(datas)
        xpointsA = [sublist[2] for sublist in datasupdate]
        ypointsA = [sublist[3] for sublist in datasupdate]
        personID = [sublist[1] for sublist in datasupdate]
        
      #  dataT=normalize_dataset(datasupdate)
        #directX = [sublist[5] for sublist in dataT]
       # directY=[sublist[6] for sublist in dataT]
       
        labelsc = assignClusterID(personID, pedid, labelsK)#memasukkan cluster ID pada frame awal(initial) ke frame saat ini, pedid=data frame awal, person id sekarang
        updatenewcluster(labelsc,datasupdate,additionalchange)#memasukkan cluster atau member yang baru ada ke dalam frame terkini
       
        l=[]
        for la in labelsc:
            l.append(la[0])
        maxcl = int(np.nanmax(l))+1
        datasnonnormal=datasupdate.copy()
        datanormal=normalize_dataset(np.array(datasupdate))
        if(frame==1010):
                a=0
        dataclust = arrangeddatas(maxcl, l, datanormal, datasupdate)
        
        # if(frame==82):
        #     a=0    
        #     for datasiu in datasupdate:
        #         if(datasiu[-1]==3):
        #             print(datasiu[-1])  #sebelum di evaluasi sudah ada  
        if (frame % 10 == 0 or frame == begining):  #evaluasi tiap 10 frame sama di awal
            if(frame!=begining):
                maxcluster=findmaxcluster(centroids)
            else:
                maxcluster=findmaxclusterbegin(datasupdate)

            labeleval = evaluatecluster(labelsc,datanormal,dataclust,datasupdate, frame, centro,maxcluster)#evaluasi cluster, disini centronya ga 
            changelabelSK(personID, pedid, labelsK, labelsc)  #mengubah labelSK

        lsc=[]
        for la in labelsc:
            lsc.append(la[0])

        maxcl=int(np.nanmax(lsc))+1
        allclust=np.unique(lsc)#disini sebelas ada
        #try:
        dataclustforcentroid = arrangeddatas(maxcl, lsc, datasnonnormal, datasupdate)#kemungkinan di penggunanan maxcluster instead of the real cluster
        centro = findcentroid(dataclustforcentroid,dataclusterprevious,frame,begining,prevcentroids,centroids)

        dataclusterprevious=dataclustforcentroid.copy()
        prevcentroids=centro.copy()
        datasupdates.append(datasupdate.copy())
        if (frame % 1 == 0 or frame == begining):
            if(frame == 85):
                a=0
            try:
                disteval.append([frame, calculateDistanceInside(centro, dataclustforcentroid,labelsc,datasupdates)])
            except Exception as qwe:
                print('no 8 ',qwe)
        
        centroids.append(centro)
        membercentroid.append(dataclustforcentroid)

    return centroids,disteval,membercentroid,maxcl
def arrangeperframe(centroids):
    centroperframe=[]
    for i in range(start,finish):
        frames=[]
        for centroid in centroids:
            for frame in centroid[1]:
                if(frame[0]==i):
                    frames.append([frame[0],centroid[0],frame[1],frame[2],frame[3],frame[4],frame[5],frame[6],frame[7]])
                    break
        centroperframe.append(frames)
    return centroperframe
def maskcentroids(centroids):
    arrcentro=arrangepercentroid(centroids)
    for centro in arrcentro:
        prevframe=[]
        for idx,frame in enumerate(centro[1]):
            if(len(prevframe)>0):
                losslen=frame[0]-prevframe[0]
                if(losslen>1):
                    avgx=avgy=avgd=avgvx=avgvy=0
                    for i in range(int(losslen)):
                        avgx+=(frame[1]-prevframe[1])/losslen
                        avgy+=(frame[2]-prevframe[2])/losslen
                        avgd+=(frame[3]-prevframe[3])/losslen
                        avgvx+=(frame[4]-prevframe[4])/losslen
                        avgvy+=(frame[5]-prevframe[5])/losslen
                        framevalue=[prevframe[0]+i+1,prevframe[1]+avgx,prevframe[2]+avgy,prevframe[3]+avgd,prevframe[4]+avgvx,prevframe[5]+avgvx,prevframe[6]+avgvy,0]

                        centro[1].insert(idx+i,framevalue)
            prevframe=frame
        prevcentroid=frame
    return arrcentro
centroids, distev,membercentro,maxclus = runtheframe(start,finish)
centroidsarranged=maskcentroids(centroids)
centroids=arrangeperframe(centroidsarranged)
distevformatted = evaluationpercluster(centroids, distev)
averagevaluepercluster=[]
averagevalueperclustermorethanonemember=[]
for disclus in distevformatted:
    avgcluster=0
    countframe=0
    countmember=0
    for frameperclus in disclus[1]:
        countframe+=1
        avgcluster+=frameperclus[1][1]
        countmember+=frameperclus[1][2]
    if(countframe>0):
        avgcluster=avgcluster/countframe
        countmember=countmember/countframe
    else:
        avgcluster=0
    averagevaluepercluster.append([disclus[0],avgcluster,countmember])
averagevaluepercluster=np.array(averagevaluepercluster)
condition=averagevaluepercluster[:,2]>1
totalaveragevaluemorethan2=np.average(np.array(averagevaluepercluster)[condition,1])
countmorethan1=len(np.array(averagevaluepercluster)[condition,1])
totalaveragevalue=np.average(np.array(averagevaluepercluster)[:,1])
findmaxdis=np.argpartition(averagevaluepercluster[:,1], -1)[-1:]

idclustermaxdis=[]
for idclus in findmaxdis:
    idclustermaxdis.append(int(distevformatted[idclus][0]))
    print('the worstnya adalah:')
    a=distevformatted[idclus]
    print(averagevaluepercluster[idclus])

print('the total average distance value is:',totalaveragevalue)
print('the total average distance value > 2 is',totalaveragevaluemorethan2)
#print('total average single cluster acrros the frame:',singlecluster)
print('total cluster created is',maxclus)
print('total cluster more than 1 is',countmorethan1)
print('percentage of cluster=',(countmorethan1/maxclus)*100)
#how many member of pedestrian in each frame based on centroids:
#centroid contains: frame,idcluster,x,y,V,Vx,Vy,jumlah cluster
pedestriannumbereveryframesfromcluster = sumclustermembereveryframes(centroids)
pedestriannumbereveryframesfromdataset = countpedestrianperframes(df,start,finish)
evsmooth=smoothnessevaluationpercluster(centroids,0.5)
evsmoothped=smoothnessevaluationperpedestrian(df,start,finish)
def findanomalouspath(evsmooth,threshold):
#find an unnatural behaviour for clusters path in its trajectory, 
#return list of anomalouspath in every cluster and 
#precentage of cluster that has anomalous err
#precentage of average cluster error length compared to its length
#percentage of cluster that has long lifespawn
#how long the error affect the total length of the entire path and occurance    
    errev=[]
    errcentro=totalpercentagepatherrorlength=totalerrorcount=0
    for ev in evsmooth:
        errcount=0
        errlength=0
        prevev=totallength=0
        for idx,evframe in enumerate(ev[1]):
            #dev=abs(evframe[-1]-prevev)
            dev=evframe[-1]
            if(dev>threshold and idx!=0):
                errcount+=1
                errlength+=dev
            prevev=evframe[-1]
            totallength+=dev
        #percentagepatherror=errcount/len(ev[1])
        if(totallength>0):
            percentagepatherrorlength=errlength/totallength
        else:
            percentagepatherrorlength=0
        
        if errcount>0:
            errcentro+=1
            totalpercentagepatherrorlength+=percentagepatherrorlength
            totalerrorcount+=errcount
           # totalpercentagepatherror+=percentagepatherror
        errev.append([ev[0],errcount,percentagepatherrorlength])
    avgcentroerr=errcentro/len(evsmooth)
    avgpercentageerrorlength=totalpercentagepatherrorlength/len(evsmooth)
    avgtotalerrorcount=totalerrorcount/len(evsmooth)
    #avgpercentageerrorcount=totalpercentagepatherror/len(evsmooth)
    return errev,avgcentroerr,avgpercentageerrorlength,avgtotalerrorcount
errsmoth,avgcentroerr,avgpercentageerrorlen,avgtotalerrorcount=findanomalouspath(evsmooth,30)
print('cluster that has anomalous path:', avgcentroerr*100,' %')
print('average value of average cluster error length compared to its length : ',avgpercentageerrorlen*100,' %')
print('average value of total anomalous path per cluster:', avgtotalerrorcount)


def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def angular_similarity(trajectory1, trajectory2):
    vectors1 = direction_vectors(trajectory1)
    vectors2 = direction_vectors(trajectory2)
    angles = [angle_between_vectors(v1, v2) for v1, v2 in zip(vectors1, vectors2)]
    return np.sum(angles)


a = 0

centroidsawal=np.array(centroids[0])
xpoints = centroidsawal[:,2]
ypoints = centroidsawal[:,3]
##################make plotss############################

############################animasi plot##################################
figA, axA = plt.subplots()
annotation=[]
for i in range(len(xpoints)):
    ann = str(str(centroidsawal[i][1])+",n="+str(centroidsawal[i][-1]))
   # annotation.append([labelsK[i][0],pedid[i],axA.annotate(ann, xy=(xpoints[i],ypoints[i]), xytext=(xpoints[i],ypoints[i]))])
    annotation.append([str(centroidsawal[i][1]),ann,axA.annotate(ann, xy=(xpoints[i],ypoints[i]), xytext=(xpoints[i],ypoints[i]))])
scat = axA.scatter(xpoints, ypoints, s=10)
###############################first plot, plot sublot x=cluster y=value per plot#############################
figDistev, axDistev = plt.subplots()
for fr in distev:
    y = np.array(fr[1])[:, 1]

    x=np.array(fr[1])[:, 0]
    
    axDistev.plot(x, y)
    #axDistev.bar(x, y)
lgnd = [row[0] for row in distev]
#print(lgnd)
axDistev.legend(lgnd)
axDistev.set_xlabel('Clusters')
axDistev.set_ylabel('Average Distance Value')
######################Second plot graph per Cluster######################################
figDistevf, axDistevf = plt.subplots()
#tampilan harusnya per cluster untuk tiap cluster
for clu in distevformatted:
    da=clu[1]
    y = [row[1][1] for row in clu[1]]
    x = [row[0] for row in clu[1]] 
    axDistevf.plot(x, y)
lgnd = [row[0] for row in distevformatted]
#(lgnd)
axDistevf.legend(lgnd,bbox_to_anchor=(-0.05, 1.02, 1,0.1), loc=3,
               ncol= 10, mode="expand", borderaxespad=0)
axDistevf.set_xlabel('Frame')
axDistevf.set_ylabel('Average Distance Value')
#########################################third plot, comparison of pedestrian count values#################################################
figpednumber, axpednumber = plt.subplots()

x=np.array(pedestriannumbereveryframesfromdataset)[:,0]
y=np.array(pedestriannumbereveryframesfromdataset)[:,1]
axpednumber.plot(x, y)
x = np.array(pedestriannumbereveryframesfromcluster)[:, 0]
y= np.array(pedestriannumbereveryframesfromcluster)[:, 1]
axpednumber.plot(x,y)
lgnd=['dataset','cluster']
axpednumber.legend(lgnd,bbox_to_anchor=(-0.05, 1.02, 1,0.1), loc=3,
               ncol= 10, mode="expand", borderaxespad=0)
axpednumber.set_xlabel('Frame')
axpednumber.set_ylabel('number of pedestrians')
#######################################Fourth plot: the difference in pedestrian count values.=================================================
figdevpednumber, axdevpednumber = plt.subplots()
x=np.array(pedestriannumbereveryframesfromdataset)[:,0]
y = np.array(pedestriannumbereveryframesfromdataset)[:, 1] - np.array(pedestriannumbereveryframesfromcluster)[:, 1]
lgnd = ['deviation']
axdevpednumber.plot(x,y)
axdevpednumber.legend(lgnd,bbox_to_anchor=(-0.05, 1.02, 1,0.1), loc=3,
               ncol= 10, mode="expand", borderaxespad=0)
axdevpednumber.set_xlabel('Frame')
axdevpednumber.set_ylabel('deviation number of pedestrians')

######################Fifth plot: deviation graph per cluster.######################################
figsmoothevf, axsmoothevf = plt.subplots()

for clu in evsmooth:
    da=clu[1]
    y = [row[1] for row in clu[1]] #x , dan y nya harusnya isinya per cluster
    x = [row[0] for row in clu[1]]
    axsmoothevf.plot(x, y)
    #perbaiki xynya
lgnd = [row[0] for row in evsmooth]
#print(lgnd)
# axsmoothevf.legend(lgnd,bbox_to_anchor=(-0.05, 1.02, 1,0.1), loc=3,
#                ncol= 10, mode="expand", borderaxespad=0)
axsmoothevf.set_xlabel('Frame')
axsmoothevf.set_ylabel('Distance Deviation')
################################################Sixth plot: deviation per pedestrian######################################

figsmoothpedevf, axsmoothpedevf = plt.subplots()

for clu in evsmoothped:
    da=clu[1]
    y = [row[1] for row in clu[1]] 
    x = [row[0] for row in clu[1]]
    axsmoothpedevf.plot(x, y)

lgnd = [row[0] for row in evsmoothped]

axsmoothpedevf.set_xlabel('Frame')
axsmoothpedevf.set_ylabel('Pedestrian Distance Deviation')

#######################################################Seventh plot: cluster comparison#####################################
figdetails, axdetails = plt.subplots()
clusterinput=idclustermaxdis
centroidtrack=[]
allcentroidtracks=[]
#cari cluster dengan kondisi terburuk
pedesclust=getpedistriandatafromcluster(clusterinput)
plt.xlim(0, 2500)
plt.ylim(0, 700)
for peds in pedesclust:
    for ped in peds.values():
      #  print('pedestriannya:')
     #   print(ped[0][1])
      #  print(ped[:, 2])
       # print(ped[:, 3])
        axdetails.plot(ped[:, 2], ped[:, 3],color='black')
        a=ped[0]
        
        b=ped[-1]
        axdetails.arrow(b[2], b[3], -b[5], -b[6], head_width=8, head_length=9, fc='k', ec='k')
        axdetails.annotate(str(b[0]), xy=(b[2], b[3]), xytext=(b[2], b[3]))
        axdetails.annotate(str(a[0]), xy=(a[2], a[3]), xytext=(a[2], a[3]))

for inputs in clusterinput:
    for centros in centroids:
        for centro in centros:
            if(centro[1]==inputs):
                centroidtrack.append(centro)
    allcentroidtracks.append(centroidtrack)
for centros in allcentroidtracks:
    centros=np.array(centros)
    axdetails.plot(centros[:,2],centros[:,3],color='red')
    a=centros[0]
   
    b=centros[-1]
    for cen in centros:    
        axdetails.arrow(cen[2], cen[3], -cen[5], -cen[6], shape='full', head_width=0.09, head_length=0.09, length_includes_head=False, color='r')
    axdetails.annotate(str(a[0]), xy=(a[2], a[3]), xytext=(a[2], a[3]))
    axdetails.annotate(str(b[0]), xy=(b[2], b[3]), xytext=(b[2], b[3]))
#cari yang deviationnya paling jelek trus selidiki,
#buat perhitungan persentasenya
#buat revoking datasetnya# tapi kerjakan setelah buat manuscript, see how they can handle this 
#i=0


###############################cluster simulation#################################################
figallcentro, axallcentro = plt.subplots()
arrangedcentroid=arrangepercentroid(centroids)
i=0
for centros in arrangedcentroid:
    #centros=np.array(centros)
    #clu=centros[1]
    y = [row[2] for row in centros[1]]
    x = [row[1] for row in centros[1]]

    axallcentro.plot(x,y,color='red')
    a=centros[1][0]
    b=centros[1][-1]
   # for cen in centros:    
    axallcentro.arrow(b[1], b[2], -b[4], -b[5], shape='full',width=0.0001, head_width=1, head_length=1, length_includes_head=True, color='black')
    axallcentro.annotate(str(centros[0]), xy=(a[1], a[2]), xytext=(a[1], a[2]))
  #  axallcentro.annotate(str(b[0]), xy=(b[1], b[2]), xytext=(b[1], b[2]))

import numpy as np
import csv

# Create a sample NumPy array

# Specify the filename

#filename = "row_by_row_output.csv"

# Open the file in write mode
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write each row to the CSV file
    for frames in arrangedcentroid:
        for row in frames[1]:
            row=np.insert(row, 1, frames[0])

            writer.writerow(row)

print(f"Data has been written to {filename}")

#####animation#####################
def update(frame):

    frame=frame+start
    #print(frame)
    # for each frame, update the data stored on each artist.
    dataA = getdatacentroidframe(centroids,frame)
    
    labelsMark = []
    labelscolors1=[]
    datasupdate = np.array(dataA)
    
    #datas=normalize_dataset(datas)
    xpointsA = [sublist[2] for sublist in datasupdate]
    ypointsA = [sublist[3] for sublist in datasupdate]
    personID = [sublist[1] for sublist in datasupdate]
    numberN = [sublist[-1] for sublist in datasupdate]
  #  print('---check---')
   # print(personID)
  #  print('annot')
  #  print(np.array(annotation)[:,0])
    try:
        for idx, d in enumerate(datasupdate):
            convert = np.array(annotation)[:, 0]
            #convertfloat=convert[1].split(",")[0]
            foundya = np.where(convert == str(d[1]))
            if (len(foundya[0]) == 0):
                lblnew=str(str(d[1])+",nu= "+str(d[-1]))
                annotation.append([str(d[1]), lblnew, axA.annotate(lblnew, xy=(300, 300), xytext=(300, 300))])


        for num, annot in enumerate(annotation):
            #anotfloat=annot[1].split(",")
            idx = idxreturncluster(float(annot[0]), personID)
            
            # if (float(annot[1]) == 41):
            #     a=0
            if (len(idx) > 0):
                annot[2].set_position((xpointsA[idx[0]], ypointsA[idx[0]]))  #coba ubah annotation cluster biar kelihatan cuk
                labeldirvel=str(personID[idx[0]])+str(numberN[idx[0]])#+ '/' + "{:.2f}".format(directX[idx[0]])+'-'+ "{:.2f}".format(directY[idx[0]])
                annot[2].set_text(labeldirvel)#str(labelsK[idx[0]][0])+"/"+str(labelidx[0][0]) )
            else:
                annot[2].set_text("-99")

    except Exception as e:
       # print('no 9 ',e)
        a=9

    
    dataAn = np.stack([xpointsA, ypointsA]).T
    scat.set_offsets(dataAn)
    #scat.set_color(labelscolors1)
    return (scat,annotation)


#centroids.append([clus[0][0], clus[0][-1], x, y, V, Vx, Vy])  #frame,idcluster,x,y,V,Vx,Vy


ani = animation.FuncAnimation(fig=figA, func=update, frames=finish-start, interval=100)

plt.show()      

#print(res.labels_)

# Background color