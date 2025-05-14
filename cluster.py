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
#preprocessing

def countcoordinatevelocity(xorycenters1,xorycenters2,length):
    return (xorycenters1 - xorycenters2) / length

def countvelocity2(xcenters, ycenters,prevVx,prevVY):#ambil data yang lama untuk diakumulasikan
    x = countcoordinatevelocity(xcenters[0], xcenters[-1],len(xcenters) )
    y = countcoordinatevelocity(ycenters[0], ycenters[-1], len(ycenters))
    xlen = (prevVx+ (xcenters[-2] - xcenters[-1]))/2
    ylen = (prevVY + (ycenters[-2] - ycenters[-1])) / 2
    if (math.isnan(xlen)):
        xlen = 0
    if (math.isnan(ylen)):
        ylen = 0
    
    return math.sqrt((x** 2) + (y** 2)),xlen,ylen

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
    # if (x < 0 and y>=0):
    #     res += 180
    # elif (x < 0 and y < 0):
    #     res += 180
    # elif (x >= 0 and y < 0):
    #     res+=360
    # return res

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
#directory='gt04A.txt'
directory='Pedestrian Cluster/gt04A.txt'
#directory='Pedestrian Cluster/pred3.txt'
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
        data[4], data[5], data[6] = countvelocity2(xcenters, ycenters, prevVx, prevVy)
        prevVx = data[5]
        prevVy = data[6]
        if (math.isnan(data[5])):
            a=0
        data[7]= countvectordirection(xcenters[-2],xcenters[-1],ycenters[-2],ycenters[-1])
    i += 1
    before += 1



datas = getdatabyframe(df, 75)
datanframe = getdatabynframe(df, 75, 100)
datasn = np.array(datanframe)
def normalize_dataset(datasn):
    datasn[:, 2] = datasn[:, 2] / datasn[:, 2].max()
    datasn[:, 3] = datasn[:, 3] / datasn[:, 3].max()
    datasn[:, 4] = datasn[:, 4] / datasn[:, 4].max()
    datasn[:, 5] = datasn[:, 5] / datasn[:, 5].max()
    datasn[:, 6] = datasn[:, 6] / datasn[:, 6].max()
    return datasn
datasn=normalize_dataset(datasn)
datasnplot = reshapeperidped(datasn)

datas=np.array(datas)
datafit = datas[:, [5,6]]

bandwidth = estimate_bandwidth(datafit, quantile=0.2, n_samples=100)
clustersms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

clusters = AgglomerativeClustering(n_clusters=20, linkage='complete', metric='manhattan')
clustersK = KMeans(n_clusters=20,init='k-means++',max_iter=250)
resK = clusters.fit(datafit)
res= clustersK.fit(datafit)
#resK= clustersms.fit(datafit)
#plt.gca().invert_yaxis()

xpoints = [sublist[2] for sublist in datas]
ypoints = [sublist[3] for sublist in datas]
vx = datas[:, 5]
vy = datas[:, 6]
pedid=[sublist[1] for sublist in datas]

def arrangedatapercluster(idclusters, lbls, datafit,dataupdate):
    #mengurutkan data percluster, jika idcluster ditemukan pada lbls maka dimasukan ke arranged
    arranged = []
    i=0
    for l in range(0, len(lbls)):
        if (idclusters == lbls[l]):
            val = datafit[l]
            val[-1] = idclusters
            dataupdate[l][-1]=idclusters
            val2=np.append(val,i)
            arranged.append(val2)#tambahkan informasi index induknya
        i+=1
    return arranged

def arrangeddatas(maxclusters, lbls, datafit, dataupdate):
    #maximum cluster, labels, data yang akan diarrange, data induk yang nilai label clusternya akan diupdate
    result=[]
    for idcluster in range(0, maxclusters):
        result.append([idcluster, arrangedatapercluster(idcluster, lbls, datafit,dataupdate)])
    return result

#def createSubcluster(labels, datafit,clustertype):
maxclus=max(resK.labels_)+1
hasil = arrangeddatas(maxclus, resK.labels_, datas,datas)
u = 0
subclusters=[]
for sub in hasil:
    #subclusters = []
    sresk=None
    if(len(sub[1])>2):
        sclustersK = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='manhattan')
        sdata=np.array(sub[1])
        sresK = sclustersK.fit(sdata[:, [2, 3]])
    else:
        sresK=None

    subclusters.append([sub[0],sresK])
#xy=np.column_stack((xpoints, ypoints))
#plt.ylim(max(plt.ylim()), min(plt.ylim()))
# fig, ax = plt.subplots()
# fig2, axK = plt.subplots()
# figMS, axMS = plt.subplots()
# figev, axev = plt.subplots()
# figev2, axev2 = plt.subplots()
# figdir,axdir=plt.subplots()
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
    for h in range(0, len(hasil)):#hasil adalah data arrange
        hasilke = np.array(hasil[h][1])[:,1]#ambil aray data idnya
        cari = np.where(d[1] == hasilke)#cari nilai id pada dnya ada atau enggak
        if (len(cari[0]) > 0):#jika datanya ditemukan maka selesai loopingnya
            break
    try:    
        subke = subclusters[h]
        if (subke[1] == None):
            slabelsK.append([subke[0],-1])    
        else:
            slabelsK.append([subke[0],subke[1].labels_[cari][0]])
        #cari label per id pedestriannya
    except Exception as E:
        s=subke
slabelsK=np.array(slabelsK)
uniqlabel=slabelsK[np.unique(slabelsK[:,[0, 1]], axis=0, return_index=True)[1]]
idx=0
for slbl in slabelsK:
    index=np.where((slbl[0]==uniqlabel[:,0]) & (slbl[1]==uniqlabel[:,1]))
    labelsK.append([index[0][0],[slbl]])

lbl=0
labelsCsK=[]
for a in labelsK:
     labelsCsK.append(colordict[int(a[0])])

# ax.quiver(xpoints, ypoints, vx, vy, color=labels, scale=10, scale_units='inches')
# axK.quiver(xpoints, ypoints, vx, vy, color=labelsK, scale=10, scale_units='inches')
# axMS.quiver(xpoints, ypoints, vx, vy, color=labelsK, scale=10, scale_units='inches')

#sipp kari dicluster per clusters. 


# for i in xpoints:
#     ax.annotate(datas[u][1], (xpoints[u], ypoints[u]))
#     u += 1


#ax.axis([-1.5, 1.5, -1.5, 1.5])
# ax.set_facecolor('lavender')
EvKmeans = []
EvHierar = []
EvMS = []
EvCKmeans = []
EvCHierar = []
EvCMS = []

# for i in range(10):
#     res = clusters.fit(datafit)
#     resK = clustersK.fit(datafit)
# #    resMS=clustersms.fit(datafit) 
# #    EvKmeans.append(silhouette_score(datafit, resK.labels_))
#     EvHierar.append(silhouette_score(datafit, res.labels_))
#     EvMS.append(silhouette_score(datafit, resMS.labels_))
#     EvCKmeans.append(calinski_harabasz_score(datafit, resK.labels_))
#     EvCHierar.append(calinski_harabasz_score(datafit, res.labels_))
#  #   EvCMS.append(calinski_harabasz_score(datafit, resMS.labels_))
    #time.sleep(0.5)

# for pedid,pedarr in datasnplot:
#     axdir.plot(pedarr[:, [2]], pedarr[:, [3]])
#     if(pedarr.size>0):
#         a = pedarr[-2]
#         x = a[2]
#         y = a[3]
#         476
#         if (pedid == 476):
#             s=10
#         plt.arrow(a[2], a[3], -a[5], -a[6], shape='full', lw=2,
#             length_includes_head=True, head_width=3, color='r')
#         axdir.annotate(pedid, (a[2], a[3]))
        
#axdir.plot(datasn[0][1])
# axev.plot(EvCKmeans,color='r')
# axev.plot(EvCHierar, color='blue')
# axev.plot(EvCMS,color='green')
# axev.set_title('Distance between cluster membership')
# axev2.plot(EvKmeans, color = 'r')
# axev2.plot(EvHierar, color='black')
# axev2.plot(EvMS,color='green')
# axev2.set_title('Distance Between Cluster')

figA, axA = plt.subplots()
annotation=[]
for i in range(len(xpoints)):
    labelidx=labelsK[i][1]
    ann=str(labelsK[i][0])+"/"+str(labelidx[0][0])
    annotation.append([labelsK[i][0],pedid[i],axA.annotate(ann, xy=(xpoints[i],ypoints[i]), xytext=(xpoints[i],ypoints[i]))])


#line2=axA.plot(xpoints, ypoints)[0]
#line2= axA.quiver(xpoints, ypoints, vx, vy, color=labels, scale=10, scale_units='inches')
scat = axA.scatter(xpoints, ypoints, color=labelsCsK, s=10)
# for i in range(len(xpoints)):
#     scat=axA.scatter(xpoints[i], ypoints[i], color='black', marker=labelsMarker[i])

#problem: ga bisa dapet centroid sebagai perwakilan cluster untuk di masukkan ke dalam perhitungan either menggunakan hitungan average, or median, or modus
#opsi: bisa menggunakan k-means clustering tapi masalahnya kmeans hasilnya bisa beda beda, bisa dapat centroid, menggunakan dbscan jg bisa : masalahnya ga ada centroid juga, konsepnya berbasis density

def idxreturnpedestrian(anno,personID):
    #mencari posisi yang sesuai dengan annotation
    return np.where(personID == anno)
def ev(member, isdist,threshold,frame,clusterno):
    #return true if there are outilers, return false when no outlier
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor
    from scipy.spatial import distance_matrix, distance
    
    if(isdist):
        dist = member[:, (2,3)]
    else:
        dist =member[:, (2, 3)]
    #dist =np.array([[0.76851957, 0.27364262], [0.67166468, 122.91234101]])#, [0.77763776, 0.2100406 ], [0.78488701, 0.10901336], [0.92527987, 0.13276037]])
    # dm = distance_matrix(dist,dist)  #get distance matrix
    
    # pointsavg=[]
    # for point in dm:
    #     avg=np.sum(point)
    #     pointsavg.append(avg)
    # distbase = np.max(pointsavg) - np.average(pointsavg)
    #menggunakan teknik LOF
    if (frame > 200):
        if (clusterno == 5):
            a=0
    if(len(member)>2):
        status=False
        if (np.isnan(dist).any() == False):  #jika mencapai maks nilai vector arah diganti 0
            if (len(member) < 4):
                n = int(len(member)) - 1
                if (clusterno == 25):
                    a=0
            else:
                if (clusterno == 25):
                    n = int(len(member)) -2 
                else:
                    n = int(len(member)) - 2
            #x = [dist[:, 0]]
            #y = [dist[:, 1]]
            #print(np.shape(x))
            #clf=distance_matrix(dist,dist)
            clf = LocalOutlierFactor(n_neighbors=n,p=2)
            result = clf.fit_predict(dist)
            resultneg = clf.negative_outlier_factor_
            # for idx,r in enumerate(resultneg):
            #     if (r - (sum(resultneg)/len(resultneg)) > 0.3):
            #         result[idx] = -1
            #     else:
            #         result[idx] = 1
                
            rest = np.where(result == -1)
            if (len(rest[0]) > 0):
                status=True
        else:
            rest = [[1]]
    elif (len(member)==2):
        dista = distance.euclidean(dist[0], dist[1])
        #berinilai nan karena data pertama jadi masih kosong.
        #cluster nan masih masuk ini adalah data baru 
        if (dista > threshold):
            rest = [[1]]
            status = True
        else:
            rest = []
            status=False
    return rest,status 
    # for m in member:
    #     distance.euclidean()
def findnearestradius(ped, datanormal,threshold):
    #mencari kandidat tetangga yang masih didalam radius
    from scipy.spatial import distance_matrix,distance
    neighbors = []
    distances = []
    directions=[]
    clusternear = []
    idx = []
    i = 0
    #a = res.predict([ped[5], ped[6]])
    #ubah sini biar yang dihitung centroidnya
    for datan in datanormal:
        pointA = [ped[2],ped[3]]
        pointB = [datan[2], datan[3]]
        pointAdir =[ped[5],ped[6]]
        pointBdir = [datan[5], datan[6]]
        distadir=distance.euclidean(pointAdir,pointBdir)
        dista = distance.euclidean(pointA, pointB)
        if(distadir<threshold):
            if (dista < threshold):
                if(datan[1]!=ped[1] and not(math.isnan(datan[-1]))):
                    neighbors.append(datan)
                    distances.append(dista)
                    directions.append(distadir)
                    idx.append(i)
        i += 1
    distances = np.array(distances)
    directions=np.array(directions)
    accumulativedistance=np.array([[]])
    try:
        if (len(neighbors) > 0):
            normaldistances = preprocessing.normalize([distances])
            normaldirections=preprocessing.normalize([directions])
            accumulativedistance=np.add(normaldistances,normaldirections)#melakukan penjumlahan distance direction dalam kondisi normal
            clusternear = np.array(neighbors)[:, -1]  #mengindikasikan bahwa nomor cluster
    except Exception as e:
        print(e)
    #mengembalikan data tetangga, akumulative distancenya, id cluster terdekatnya dan id data tetangganya
    return neighbors,accumulativedistance[0],clusternear,idx
def evaluatecluster(label,datanormal,dataclust, datasupdate,frame):
    #main cluster evaluation, urutkan dulu berdasarkan cluster kemudian lakukan evaluasi jaarak dan arah per clusternya
    # l=[]
    # for la in label:
    #     l.append(la[0])
    # maxcl = len(np.unique(l))
    # datanormal=normalize_dataset(np.array(datasupdate))
    # dataclust = arrangeddatas(maxcl, l, datanormal,datasupdate)  #mengurutkan data cluster sesuai dengan urutan frame
    b = datasupdate
    thres=0.1
    ix = []
    for percluster in dataclust:
        if (len(percluster[1]) > 1):#what if less than 2
            member = np.array(percluster[1])
            disevres,disstat = ev(member, False,3,frame,percluster[0])
            #direvres, dirstat = ev(member, False,0.5)
            
            if (disstat == True):
                for idex in disevres:
                    #tunjukan perubahaannya
                    change = member[idex]
                    if (len(idex) == 1):
                        try:
                            #change = member[idex]
                            neighbors, dist, clustnear, indexofresult = findnearestradius(change[0], datanormal, thres)
                            #cari radius yang dekat bagi yang clusternya berubah
                            if(len(dist)>0):
                                val = neighbors[np.argmin(np.array(dist))][-1]
                                indexneighbors=int(indexofresult[np.argmin(np.array(dist))])
                                vallabel= label[indexneighbors]
                                datasupdate[int(change[0][-1])][-1] = vallabel[0]
                                label[int(change[0][-1])] = vallabel
                            #label[int(change[0][-1])][0]=11#konek nih sayyyyyy
                        except Exception as e:
                                print(e)
                    else:
                        for c in change:
                            try:
                                neighbors, dist, clustnear,indexofresult = findnearestradius(c, datanormal, thres)
                                #ini gw cuman cari jarak terdekat terus lgs diassign
                            # print(int(change[-1]))
                                if(len(dist)>0):
                                    idxdis=np.argmin(np.array(dist))
                                    val = neighbors[np.argmin(np.array(dist))][-1]
                                    indexneighbors=int(indexofresult[np.argmin(np.array(dist))])
                                    vallabel= label[indexneighbors]
                                    datasupdate[int(c[-1])][-1] = vallabel[0]#int(-1)
                                    label[int(c[-1])] = vallabel
                                #label[int(change[0][-1])][0]=11
                            except Exception as e:
                                print(e)
                            
                    #label[int(change[-1])][0]=
                  #  print(datas[int(change[0][-1])])
                  #  ix.append([int(change[-1]), val])
                    #ax=3/0
                    #find nearest by threshold radius
                    b = datasupdate[idex]
                    #make grid per frame
                    #find minimum
                    #if larger than threshold assign it!
    nancluster=[]
    for idx,data in enumerate(datanormal):
        cluster = data[7]
        if (math.isnan(data[7])):
                neighbors, dist, clustnear, indexofresult = findnearestradius(data, datanormal, thres)       
                if (len(neighbors) == 0):
                    nancluster.append(data)
                else:
                    try:
                        idxmin=np.argmin(dist)
                        label[idx] = [clustnear[idxmin], -1]#pilih yang paling deket jarak dan directionnya
                        datasupdate[idx][-1] = clustnear[idxmin]
                        additionalchange.append(datasupdate[idx])
                        a = 0
                    except Exception as e:
                        print(e)

                    #jadikan cluster tersebut

                #dijadikan centroid kah?
                #apakah akan di clusterkan berdasarkan posisi menggunakan kmeans? let sett
    
    if(len(nancluster)>5):#jika cluster yg tidak ditemukan nearest radiusnya lebih dari 5 maka
        nanclusters = np.array(nancluster)
        predictedval=nanclusters[:,[5,6]]
        resultnewclust = res.predict(predictedval)  #lakukan prediksi menggunakan kmeans untuk direksinya
        resultnewclustunique = np.unique(resultnewclust)
        nangroup = []
        for nan in resultnewclustunique:#mengelompokan data baru berdasarkan cluster baru
            nansubgroup=[]
            for idxes,nancluster in enumerate(nanclusters):
                if (resultnewclust[idxes]==nan):
                    nansubgroup.append(nancluster)
            nangroup.append(nansubgroup)
        #subclusternew=[]
        for nanfit in nangroup:#melakukan pengclusteran terhadap cluster, -1 untuk tidak ada subcluster, 0 untuk cluster pertama dan 1 untuk cluster ke dua
            if(len(nanfit)>1):
                agglo = AgglomerativeClustering(n_clusters=2)
                rescluster = agglo.fit(np.array(nanfit)[:, [2, 3]])
                labelsrescluster=rescluster.labels_
                lastclusternumber = np.nanmax(datasupdate[:, -1])
                labelsrescluster[labelsrescluster == 0] = lastclusternumber + 1
                labelsrescluster[labelsrescluster == 1] = lastclusternumber + 2
                a = 0
                for i, lc in enumerate(labelsrescluster):
                    dataid = datasupdate[:, 1]
                    idtosearch=nanfit[i][1]
                    index = np.where(dataid==idtosearch)[0][0]
                    datasupdate[index][-1] = lc
                    label[index] = [lc, -a]
                    additionalchange.append(datasupdate[index])
                #lakukan assigne0ment pada cluster baru yang mengubah nilai label dan nilai datasupdate
            else:
                
                lastclusternumber = np.nanmax(datasupdate[:, -1])
                rescluster = lastclusternumber + 1
                dataid = datasupdate[:, 1]
                idtosearch=nanfit[0][1]
                index = np.where(dataid==idtosearch)[0][0]
                datasupdate[index][-1] = rescluster

                label[index] = [rescluster, -900]
                additionalchange.append(datasupdate[index])

                #lakukan hal yang sama lgs dilkaukan assignement cluster pada nilai label dan nilai datasupdate
            #subclusternew.append(rescluster)
        
        # for idx,d in enumerate(datasupdate):
        #     for nanfit in nangroup:
        #         for nandata in nanfit:
        #             if (d[1] == nandata[1]):
        #                 a=0
                        #d[-1]=np.max(np.unique(labels))+1

        #lakukan pencarian kembali index pada data sebelumnya untuk mengclusterkan        

    return ix
wrong = []
def assignClusterID(searchpersonIDs, personIDcluster, clusterlabels):
    #mengurutkan cluster berdasarkan urutan inputnya
    cluster=[]
    for pID in searchpersonIDs:
        try:
            find = personIDcluster.index(pID)
            cluster.append(clusterlabels[find])
        except ValueError:
            cluster.append([-1,-1])
    return cluster
def changelabelSK(databaru, datalama, labellama,labelbaru):
    #diketahui clusterID, data lama, data baru
    #mau mengubah cluster IDnya
    try:
        i = 0
        a=0
        for db in databaru:
            find = -1
         
            try:
                find = datalama.index(db)
                #if (labellama[find][0] == 25):
               #     a+=1
            except Exception as e:
                find=-1
            if (find >= 0):
                labellama[find] = labelbaru[i]
            i += 1
        a=0
    except Exception as e:
        print(e)
def updatenewcluster(labelsc, datasupdate, additionalchange):
    for idx,du in enumerate(datasupdate):
        for ac in additionalchange:
            if (du[1] == ac[1]):
                du[-1] = ac[-1]
                labelsc[idx]=[ac[-1],-1]    
def findcentroid(datapercluster):
    centroids=[]
    for clus in datapercluster:
        clus = np.array(clus[1])
        n = len(clus)
        try:

            if(n>1):
                x = sum(clus[:, 2])/n
                y = sum(clus[:, 3])/n
                V=sum(clus[:,4])/n
                Vx = sum(clus[:, 5])/n
                Vy=sum(clus[:,6])/n
                centroids.append([clus[0][0], clus[0][-1], x, y, V, Vx, Vy])  #frame,idcluster,x,y,V,Vx,Vy
            else:
                centroids.append([clus[0][0], clus[0][-1], clus[0][2], clus[0][3], clus[0][4], clus[0][5], clus[0][6]])
        except Exception as E:
            a=clus
            
    return centroids
additionalchange = []
def runtheframe(begining):
    maxframe = max(np.array(df)[:, 0])
    centroids=[]
    for frame in range(begining,maxframe):
        dataA = getdatabyframe(df,frame)
        labelsMark = []
        labelscolors1=[]
        datasupdate = np.array(dataA) 
        #datas=normalize_dataset(datas)
        xpointsA = [sublist[2] for sublist in datasupdate]
        ypointsA = [sublist[3] for sublist in datasupdate]
        personID = [sublist[1] for sublist in datasupdate]
        dataT=normalize_dataset(datasupdate)
        directX = [sublist[5] for sublist in dataT]
        directY=[sublist[6] for sublist in dataT]
        
        labelsc = assignClusterID(personID, pedid, labelsK)#memasukkan cluster ID pada frame awal ke frame saat ini
        updatenewcluster(labelsc,datasupdate,additionalchange)#memasukkan cluster atau member yang baru ada ke dalam frame terkini
        l=[]
        for la in labelsc:
            l.append(la[0])
        maxcl = len(np.unique(l))
        datanormal=normalize_dataset(np.array(datasupdate))
        dataclust = arrangeddatas(maxcl, l, datanormal, datasupdate)
        
        if(frame%10==0 or frame==75):
            labeleval = evaluatecluster(labelsc,datanormal,dataclust,datasupdate, frame)
            changelabelSK(personID, pedid, labelsK, labelsc)
        
        maxcl = int(np.nanmax(datasupdate[:,-1])+1)
        dataclustforcentroid = arrangeddatas(maxcl, l, datasupdate, datasupdate)
        centroid = findcentroid(dataclustforcentroid)
        centroids.append(centroid)
    return centroids
#centroids = runtheframe(75)
a=0
def update(frame):
    #masalah label kosong aja
    frame=frame+75
    # for each frame, update the data stored on each artist.
    dataA = getdatabyframe(df,frame)
    labelsMark = []
    labelscolors1=[]
    datasupdate = np.array(dataA)
    
    #datas=normalize_dataset(datas)
    xpointsA = [sublist[2] for sublist in datasupdate]
    ypointsA = [sublist[3] for sublist in datasupdate]
    personID = [sublist[1] for sublist in datasupdate]
    dataT=normalize_dataset(datasupdate)
    directX = [sublist[5] for sublist in dataT]
    directY=[sublist[6] for sublist in dataT]
    
    labelsc = assignClusterID(personID, pedid, labelsK)#memasukkan cluster ID pada frame awal ke frame saat ini
    updatenewcluster(labelsc,datasupdate,additionalchange)#memasukkan cluster atau member yang baru ada ke dalam frame terkini
    l=[]
    for la in labelsc:
        l.append(la[0])
    unique=np.unique(l)
    maxcl = len(np.unique(l))
    #maks=np.nanmax(datasupdate[:,-1])
    datanormal=normalize_dataset(np.array(datasupdate))
    dataclust = arrangeddatas(maxcl, l, datanormal, datasupdate)
    
    if(frame%10==0 or frame==75):
        labeleval = evaluatecluster(labelsc,datanormal,dataclust,datasupdate, frame)
        changelabelSK(personID, pedid, labelsK, labelsc)
    
    #unique=np.unique(l)
  
    # cari = personID.index(511)
    # clusternya = labelsc[cari]
    # if (clusternya[0] == 5):
    #     a = 0
    # elif(clusternya[0] ==11):
    #     b=0
    
    # cari = personID.index(511)
    # clusternya = labelsc[cari]
    # if (clusternya[0] == 5):
    #     a = 0
    # elif(clusternya[0] ==11):
    #     b=0
    #ubah label SKnya

    #wrong.append(labeleval)
    #print(wrong)
    #labelsc[labeleval[0][0]] = labeleval[0][1]#mengubah nilai labelnya ke nilai yang bersesuaiann, cek kesesuaian perubahan ini, masih ada bugs, karena semua jadi ijo cuk
    #menguba
    #evaluate cluster score based on direction
    #annotation.set_position(-xpointsA,-ypointsA)
    it = 0
    # for i in range(len(xpoints)):
    # labelidx=labelsK[i][1]
    # ann=str(labelsK[i][0])+"/"+str(labelidx[0][0])    
    # annotation.append([labelsK[i][0],pedid[i],axA.annotate(ann, xy=(xpoints[i],ypoints[i]), xytext=(xpoints[i],ypoints[i]))])
    try:
        for d in datasupdate:#ini untuk mencari yang label pada data updatesnya masih nan akan ditampilkan sebagai -1
            #for i in range(len(xpoints)):
            #labelidx=labelsK[i][1]
            if (math.isnan(d[-1])):
                convert=np.array(annotation)[:,1]
                foundya = np.where(convert == d[1])#mencari id di datasupdate(data yang sedang berjalan di loop) dari anotasi, 
                if(len(foundya[0])==0):#jika id tidak ditemukan maka nilainya dijadikan -1 (ini niatnya menargetkan data data nan untuk ditampilkan jadi -1)
                    ann="-1"
                    annotation.append(['-1', d[1], axA.annotate(ann, xy=(d[2], d[3]), xytext=(d[2], d[3]))])
            else:
                convert=np.array(annotation)[:,1]
                foundya = np.where(convert == d[1])
                if (len(foundya[0]) == 0):
                    lblnew="p"+str(d[-1])
                    annotation.append([lblnew, d[1], axA.annotate(lblnew, xy=(d[2], d[3]), xytext=(d[2], d[3]))])
        # a=0   

        for num, annot in enumerate(annotation):
            idx = idxreturnpedestrian(annot[1], personID)[0]
            
            if (len(idx) > 0):
                annot[2].set_position((xpointsA[idx[0]], ypointsA[idx[0]]))  #coba ubah annotation cluster biar kelihatan cuk
                # if (annot[0] != "-1"):
                #     cluster2=labelsc[idx[0]][1][0]
                #     labelidx = str(labelsc[idx[0]][0]) + '/' + str(cluster2[0])
                #print()
                if(labelsc[idx[0]][0]==999):
                    labeldirvel='anjayyyy!!!'
                else:
                    labeldirvel="p"+str(personID[idx[0]])+str(labelsc[idx[0]][0])#+ '/' + "{:.2f}".format(directX[idx[0]])+'-'+ "{:.2f}".format(directY[idx[0]])
                annot[2].set_text(labeldirvel)#str(labelsK[idx[0]][0])+"/"+str(labelidx[0][0]) )
            else:
                annot[2].set_text(" ")
                #a=0
                #annotation.remove(annot)
                #annot[2].set_position((0, 0))
            it+=1
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
            labelscolors1.append(colordict[int(a[0])])
    except Exception as e:
        print(e)
    dataAn = np.stack([xpointsA, ypointsA]).T
    scat.set_offsets(dataAn)
    scat.set_color('black')#labelscolors1)
    
    #scat.set_marker('x')
    
    # for i in range(len(xpointsA)):
    #     scat=axA.scatter(xpointsA[i], ypointsA[i], color='black', marker=labelsMark[i])
    #scat = axA.scatter(xpointsA, ypointsA,c='black',s=5)
    #scat.set_color(black)

    return (scat,annotation)



ani = animation.FuncAnimation(fig=figA, func=update, frames=1000, interval=50)

plt.show()      
#buat clusteringnya cuys
print(res.labels_)

# Background color
