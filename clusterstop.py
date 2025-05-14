import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

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
#preprocessing

def countcoordinatevelocity(xorycenters1,xorycenters2,length):
    return (xorycenters1 - xorycenters2) / length


def countvelocity(xcenters, ycenters):
    x = countcoordinatevelocity(xcenters[0], xcenters[-1], len(xcenters))
    y = countcoordinatevelocity(ycenters[0], ycenters[-1], len(ycenters))
    xlen = xcenters[-2] - xcenters[-1]
    ylen=ycenters[-2]- ycenters[-1]
    return math.sqrt((x** 2) + (y** 2)),xlen,ylen
def countvectordirection(x1, x2, y1, y2):

    x = x1 - x2
    y = y1 - y2
    res = math.degrees(math.atan(y / x))
    if (math.isnan(res)):
        res = -99
    # if (x < 0 and y>=0):
    #     res += 180
    # elif (x < 0 and y < 0):
    #     res += 180
    # elif (x >= 0 and y < 0):
    #     res+=360
    return res

def getdatabyframe(df,frameno):
    dataframe = []
    for data in df:
        if (data[0] == frameno):
            dataframe.append(data)
    return dataframe

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
directory='Pedestrian Cluster/pred04.txt'
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
databefore=-1
for data in df:
    if (i > 0):
        if(df[before][1] != data[1]):
            xcenters = []
            ycenters = []
            i=0
    xcenters.append(data[2])
    ycenters.append(data[3])
    if (i > 0):
        #databefore=df[i][1]
        if (i > 100):
            a=0
        data[4],data[5],data[6] = countvelocity(xcenters, ycenters)
        data[7]= countvectordirection(xcenters[-2],xcenters[-1],ycenters[-2],ycenters[-1])
    i += 1
    before += 1


#harusnya rerata directionnya ga sih
datas = getdatabyframe(df, 75)
datanframe = getdatabynframe(df, 75, 400)
datasn = np.array(datanframe)
datasn[:, 2] = datasn[:, 2] / datasn[:, 2].max()
datasn[:, 3] = datasn[:, 3] / datasn[:, 3].max()
datasn[:, 4] = datasn[:, 4] / datasn[:, 4].max()
datasn[:, 5] = datasn[:, 5] / datasn[:, 5].max()
datasn[:, 6] = datasn[:, 6] / datasn[:, 6].max()
#datasn[:, 7] = datasn[:, 7] / datasn[:, 7].max()
datasnplot = reshapeperidped(datasn)

datas=np.array(datas)
datafit = datas[:, [5,6]]

bandwidth = estimate_bandwidth(datafit, quantile=0.2, n_samples=100)
clustersms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

clusters = AgglomerativeClustering(n_clusters=20, linkage='complete', metric='manhattan')
clustersK = KMeans(n_clusters=16)
resK = clusters.fit(datafit)
res= clustersK.fit(datafit)
#resK= clustersms.fit(datafit)
#plt.gca().invert_yaxis()

xpoints = [sublist[2] for sublist in datas]
ypoints = [sublist[3] for sublist in datas]
vx = datas[:, 5]
vy = datas[:, 6]
pedid=[sublist[1] for sublist in datas]

def arrangedatapercluster(idclusters,lbls,datafit):
    arranged=[]
    for l in range(0, len(lbls)):
        if (idclusters == lbls[l]):
            arranged.append(datafit[l])
    return arranged
def arrangeddatas(maxclusters, lbls, datafit):
    result=[]
    for idcluster in range(0, maxclusters):
        result.append([idcluster, arrangedatapercluster(idcluster, lbls, datafit)])
    return result
#def createSubcluster(labels, datafit,clustertype):
hasil = arrangeddatas(20, resK.labels_, datas)
u = 0
subclusters = []
subdatafit=[]
for sub in hasil:
    #subclusters = []
    sresk=None
    if (len(sub[1]) > 2):
        if(len(sub[1]) > 4):
            nclus = int(len(sub[1]) / 4)
        else:
            nclus=2
        sclustersK = AgglomerativeClustering(n_clusters=nclus, linkage='complete', metric='manhattan')
        sdata=np.array(sub[1])
        sresK = sclustersK.fit(sdata[:, [2, 3]])
    else:
        sresK = None
    for datax in sdata:
        subdatafit.append(datax)
    subclusters.append([sub[0],sresK])

figdir, axdir = plt.subplots()
#fig,ax=plt.subplots()
#ax.scatter(xpoints,ypoints,c=res.labels_)
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
#masukin subcluster label ke data utamanya
for d in datas:
    cari=[]
    for h in range(0, len(hasil)):
        hasilke = np.array(hasil[h][1])[:,1]
        cari = np.where(d[1] == hasilke)
        if (len(cari[0]) > 0):
            break
        
    subke = subclusters[h]
    if (subke[1] == None):
        slabelsK.append([subke[0],-1])    
    else:
        slabelsK.append([subke[0],subke[1].labels_[cari][0]])
    #cari label per id pedestriannya
slabelsK=np.array(slabelsK)
uniqlabel=slabelsK[np.unique(slabelsK[:,[0, 1]], axis=0, return_index=True)[1]]
idx = 0
#ini buat menyesuaikan kombinasi dengan uniq labelnya, jika kolom 1 dan 2 nya sama, maka dikembalikan index dari uniq labelnya
for slbl in slabelsK:
    index=np.where((slbl[0]==uniqlabel[:,0]) & (slbl[1]==uniqlabel[:,1]))
    labelsK.append([index[0][0],slbl])

lbl=0
labelsCsK=[]
for a in labelsK:
     labelsCsK.append(colordict[int(a[0])])
u = 0
iyd=[]
for pedid, pedarr in datasnplot:  #peddarr sama labelCsK ga sesuai nilainya cek ulang
    datas=np.array(datas)
    #indexx = np.where(pedid == datas[:, 1])
   
    axdir.plot(pedarr[:, [2]], pedarr[:, [3]],color='black')
    if(pedarr.size>0):
        a = pedarr[-1]
        x = a[2]
        y = a[3]
        if (pedid == 476):
            s=10
        plt.arrow(a[2], a[3], -a[5]/100, -a[6]/100, shape='full', lw=0.1, length_includes_head=False, head_width=0.009, color='r')
    # if (len(indexx[0]) > 0):
    #     idxx = indexx[0]
    #     iyd.append(idxx[0])
    #     axdir.plot(pedarr[:, [2]], pedarr[:, [3]],color=labelsCsK[idxx[0]])
    #     if(pedarr.size>0):
    #         a = pedarr[-1]
    #         x = a[2]
    #         y = a[3]
    #         if (pedid == 476):
    #             s=10
    #         plt.arrow(a[2], a[3], -a[5]/100, -a[6]/100, shape='full', lw=0.1, length_includes_head=False, head_width=0.009, color='r')
    #         if(labelsK[idxx[0]][1][0]==0):
    #             labelsss=str(labelsK[idxx[0]][0])+"-"+str(labelsK[idxx[0]][1])#+"/"+str(round(a[5],2))+","+str(round(a[6],2))
    #             axdir.annotate(labelsss, (a[2], a[3]))
    # u += 1
axz=12
        

#problem: ga bisa dapet centroid sebagai perwakilan cluster untuk di masukkan ke dalam perhitungan either menggunakan hitungan average, or median, or modus
#opsi: bisa menggunakan k-means clustering tapi masalahnya kmeans hasilnya bisa beda beda, bisa dapat centroid, menggunakan dbscan jg bisa : masalahnya ga ada centroid juga, konsepnya berbasis density
def assignClusterID(searchpersonIDs,personIDcluster,clusterlabels):
    cluster=[]
    for pID in searchpersonIDs:
        try:
            find = personIDcluster.index(pID)
            cluster.append(clusterlabels[find])
        except ValueError:
            cluster.append(-1)
    return cluster
def idxreturnpedestrian(labelsc,anno,personID):
    #mencari posisi yang sesuai dengan annotation
    return np.where(personID==anno)

def update(frame):
    # for each frame, update the data stored on each artist.
    dataA = getdatabyframe(df,frame)
    labelsMark = []
    labelscolors1=[]
    datas=np.array(dataA)
    xpointsA = [sublist[2] for sublist in datas]
    ypointsA = [sublist[3] for sublist in datas]
    personID = [sublist[1] for sublist in datas]
    labelsc = assignClusterID(personID, pedid, labelsK)
    #annotation.set_position(-xpointsA,-ypointsA)
    try:
        for num, annot in enumerate(annotation):
            idx = idxreturnpedestrian(labelsc, annot[1],personID)[0]
            if(len(idx)>0):
                annot[2].set_position((xpointsA[idx[0]], ypointsA[idx[0]]))
            else:
                annot[2].set_position((0, 0))
            #karena cluster maka area tujuannya juga tidak bisa exact tp bisa mencangkup suatu radius tertentuu #buat evaluation function
            #coba dataset yang berbeda
            #apa yang harus dilakukan ketika ada cluster yang terpecah
            #belajar dynamic clustering sama edge based cluster
            #annot.text=labelsc[xp]
            #dicocokan xpointsAnya dengan annotation labelnya, jadi tambahkan penanda label pada anotationnya 
            #search personidnya dulu
           #  annotation[xp].set_position((xpointsA[xp],ypointsA[xp]))
          #  annotation[xp].xy = (xpointsA[xp], ypointsA[xp])
           # annotation[xp].xytext = (xpointsA[xp], ypointsA[xp])
            #annotation[xp].text = labelsc[xp]
    except Exception as e:
        print(e)
        a=9
    # for a in labelsc:
    #     labelsMark.append(markerdict[int(a)])
    try:
        for a in labelsc:
            #labelscolors1.append(a)
            if(a==-1):
                labelscolors1.append(colordict[0])    
            else:
                labelscolors1.append(colordict[int(a)])
    except:
        a=0
    dataAn = np.stack([xpointsA, ypointsA]).T
    scat.set_offsets(dataAn)
    scat.set_color(labelscolors1)
    
    #scat.set_marker('x')
    
    # for i in range(len(xpointsA)):
    #     scat=axA.scatter(xpointsA[i], ypointsA[i], color='black', marker=labelsMark[i])
    # scat = axA.scatter(xpointsA, ypointsA,c='black',s=5)
    # scat.set_color(black)

    return (scat,annotation)



#ani = animation.FuncAnimation(fig=figA, func=update, frames=200, interval=50)

plt.show()      
#buat clusteringnya cuys
print(res.labels_)

# Background color
