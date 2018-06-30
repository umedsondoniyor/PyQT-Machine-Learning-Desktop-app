from PyQt4 import QtGui
from PyQt4.uic.properties import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtGui import *
from PyQt4.QtGui import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import math
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

from makine import Ui_Dialog


class MainWindow(QtGui.QMainWindow, Ui_Dialog):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        print "__init__"
        self.RR_veriYukle1_btn.clicked.connect(self.veri1)
        self.RR_veriYukle2_btn.clicked.connect(self.veri2)
        self.RR_RUS_btn.clicked.connect(self.rus)
        self.RR_ROS_btn.clicked.connect(self.ros)
        self.RR_RveriOlustur1_btn.clicked.connect(self.olustur1_2)
        self.RR_RveriOlustur2_btn.clicked.connect(self.olustur2_2)

        self.KNN_veriYukle_btn.clicked.connect(self.knn_veri_ekle)
        self.KNN_Kumeleme_btn.clicked.connect(self.knn_veri_kumele)
        self.KNN_YeniveriYukle_btn.clicked.connect(self.knn_veri_ekle_kumele)
        
        self.Kmeans_veriYukle_btn.clicked.connect(self.veriYukleKmeans)
        self.Kmeans_Kumele_btn.clicked.connect(self.kumelemeKmeans)


        self.TT_upload_btn.clicked.connect(self.tum_veri_ekle)
        self.TT_uygula_btn.clicked.connect(self.bol_veri_tt)
        
        self.RanForest_uygula_btn.clicked.connect(self.RandomForest)
        
        self.Norm_btn_veriYukle.clicked.connect(self.dosyaYuklenormalize)
        self.Norm_btn_MinMax.clicked.connect(self.normalize_MinMax)
        self.Norm_btn_Zscore.clicked.connect(self.normalize_Z_Score)
        self.Norm_btn_Median.clicked.connect(self.normalizeMedian)


        self.pushButton_6.clicked.connect(self.parkisyonveri)
        self.pushButton.clicked.connect(self.sstbtn)
        self.pushButton_5.clicked.connect(self.RandomFogren)
        self.pushButton_3.clicked.connect(self.dstbtn)
        self.pushButton_2.clicked.connect(self.stcpbtn)
        self.pushButton_4.clicked.connect(self.butunbtn)

#************************************* RUS - ROS TAB *********************************************************

    Y=[]
    X=[]


    def olustur1_2(self):
        self.X=[]
        for i in range(int(self.RR_miktar1_lineEd.text())):
            x="{:.2f}".format(random.uniform(float(self.RR_Varalik1_1_lineEd.text()), float(self.RR_Varalik1_2_lineEd.text())))
            y="{:.2f}".format(random.uniform(float(self.RR_Varalik1_1_lineEd.text()), float(self.RR_Varalik1_2_lineEd.text())))
            self.X.append([x,y])
        self.tablo1_goster()
        
    def olustur2_2(self):
        self.Y=[]
        for i in range(int(self.RR_miktar2_lineEd.text())):
            x="{:.2f}".format(random.uniform(float(self.RR_Varalik2_1_lineEd.text()), float(self.RR_Varalik2_2_lineEd.text())))
            y="{:.2f}".format(random.uniform(float(self.RR_Varalik2_1_lineEd.text()), float(self.RR_Varalik2_2_lineEd.text())))
            self.Y.append([x,y])
        self.tablo2_goster()
    def ros(self):
        ros=[]
        if(len(self.X)>len(self.Y)):
            ros=random.sample(self.X, len(self.Y))
            for i in range(len(ros)):
                plt.plot(self.Y[i][0], self.Y[i][1], "r", markersize = 9,marker = ".",alpha=0.2)
                plt.plot(ros[i][0], ros[i][1], "g", markersize = 5,marker = "o",alpha=0.2)
        else:
            ros=random.sample(self.Y, len(self.X)) 
            for i in range(len(ros)):
                plt.plot(ros[i][0], ros[i][1], "r", markersize = 9,marker = ".",alpha=0.2)
                plt.plot(self.X[i][0], self.X[i][1], "g", markersize = 5,marker = "o",alpha=0.2)
        plt.savefig('./sonuclar/Ros.png')
        plt.show()
        
        w,h=self.RR_ROS_GV.width()-5,self.RR_ROS_GV.height()-5
        self.RR_ROS_GV.setScene(self.show_image('./sonuclar/Ros.png',w,h))

    def rus(self):
        adet = 0
        if (len(self.X) > len(self.Y)):
            adet = len(self.X)
        else:
            adet = len(self.Y)
        for i in range(adet):
            plt.plot(self.Y[i % len(self.Y)][0], self.Y[i % len(self.Y)][1], "r", markersize=9, marker=".", alpha=0.2)
            plt.plot(self.X[i % len(self.X)][0], self.X[i % len(self.X)][1], "g", markersize=5, marker="o", alpha=0.2)
        plt.savefig('./sonuclar/Rus.png')
        plt.show()
        w, h = self.RR_RUS_GV.width() - 5, self.RR_RUS_GV.height() - 5
        self.RR_RUS_GV.setScene(self.show_image('./sonuclar/Rus.png', w, h))
        
    def tablo1_goster(self):
        if(len(self.Y)>0):
            for i in range(len(self.X)):
                plt.plot(self.X[i][0], self.X[i][1], "g",markersize = 5,marker = "o",alpha=0.2)
            for i in range(len(self.Y)):
                plt.plot(self.Y[i][0], self.Y[i][1], "r",markersize = 9,marker = ".",alpha=0.2)
        else:
             for i in range(len(self.X)):
                plt.plot(self.X[i][0], self.X[i][1], "g",markersize = 5,marker = "o",alpha=0.2)            
        plt.savefig('./sonuclar/tumveriler.png')
        plt.show()
        w,h=self.RR_TUM_VERILER_GV.width()-5,self.RR_TUM_VERILER_GV.height()-5
        self.RR_TUM_VERILER_GV.setScene(self.show_image('./sonuclar/tumveriler.png',w,h))
        self.RR_1_tabelW.setColumnCount(2)
        self.RR_1_tabelW.setRowCount(len(self.X)) ##set number of rows
        for rowNumber,row in enumerate(self.X):
            self.RR_1_tabelW.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.RR_1_tabelW.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1]))) 
    def tablo2_goster(self):
        if(len(self.X)>0):
            for i in range(len(self.X)):
                plt.plot(self.X[i][0], self.X[i][1], "g",markersize = 5,marker = "o",alpha=0.2)
            for i in range(len(self.Y)):
                plt.plot(self.Y[i][0], self.Y[i][1], "r",markersize = 9,marker = ".",alpha=0.2)
        else:
             for i in range(len(self.Y)):
                plt.plot(self.Y[i][0], self.Y[i][1], "g",markersize = 9,marker = ".",alpha=0.2)
        plt.savefig('./sonuclar/tumveriler.png')
        plt.show()
        w,h=self.RR_TUM_VERILER_GV.width()-5,self.RR_TUM_VERILER_GV.height()-5
        self.RR_TUM_VERILER_GV.setScene(self.show_image('./sonuclar/tumveriler.png',w,h))
        self.RR_2_tabelW.setColumnCount(2)
        self.RR_2_tabelW.setRowCount(len(self.Y)) ##set number of rows
        for rowNumber,row in enumerate(self.Y):
            self.RR_2_tabelW.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.RR_2_tabelW.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1]))) 

        
    def show_image(self, img_name,width,height):       
        pixMap = QtGui.QPixmap(img_name)   
        pixMap=pixMap.scaled(width,height)            
        pixItem = QtGui.QGraphicsPixmapItem(pixMap)
        scene2 = QGraphicsScene()
        scene2.addItem(pixItem)    
        return scene2    
        
    def veri1(self):
        self.veriyolu1 = unicode(QtGui.QFileDialog.getOpenFileName(self, "Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        f = open(self.veriyolu1)
        self.X=[]
        for i,row in enumerate(f.readlines()):
            currentline = row.split(",")   
            temp=[]
            for column_value in currentline:
                temp.append(column_value)
            self.X.append(temp)
            
        self.tablo1_goster()
        
            
        
    def veri2(self):
        self.veriyolu2 = unicode(QtGui.QFileDialog.getOpenFileName(self,"Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        f = open(self.veriyolu2)
        self.Y=[]
        for i,row in enumerate(f.readlines()):
            currentline = row.split(",")   
            temp=[]
            for column_value in currentline:
                temp.append(column_value)
            self.Y.append(temp)
        self.tablo2_goster()


# ********************************** RUS - ROS TAB END *********************************************************


# ********************************** KNN TAB BEGINNING *********************************************************

    Knn = []

    def knn_veri_ekle(self):
        self.fileNameKNN = unicode(
            QtGui.QFileDialog.getOpenFileName(self,"Duzenlenecek dosyayi secin", ".", "Resim dosyalari (*.*)"))
        f = open(self.fileNameKNN)
        self.Knn = []
        for i, row in enumerate(f.readlines()):
            currentline = row.split(",")
            temp = []
            for column_value in currentline:
                temp.append(column_value)
            self.Knn.append(temp)
        self.KNN_tablo_goster()

    def KNN_tablo_goster(self):
        if (len(self.Knn) > 0):
            for i in range(len(self.Knn)):
                plt.plot(self.Knn[i][0], self.Knn[i][1], "g", markersize=5, marker="o")

        plt.savefig('./sonuclar/knn/tumveriler.png')
        plt.show()

        w, h = self.KNN_veriekle_GV.width() - 5, self.KNN_veriekle_GV.height() - 5
        self.KNN_veriekle_GV.setScene(self.show_image('./sonuclar/knn/tumveriler.png', w, h))

        self.KNN_1_tabWidget.setColumnCount(2)
        self.KNN_1_tabWidget.setRowCount(len(self.Knn))  ##set number of rows
        for rowNumber, row in enumerate(self.Knn):
            self.KNN_1_tabWidget.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            self.KNN_1_tabWidget.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))

    def knn_veri_kumele(self):
        X = self.Knn
        kumeyeri = []
        kumesayisi = 2
        for i in range(len(X)):
            kumeyeri.append(int(X[i][0]) % kumesayisi)
        devammi = True
        while (devammi):
            merkezler = []
            for i in range(kumesayisi):
                merkezler.append([0, 0, 0])
            for i in range(len(X)):
                merkezler[kumeyeri[i]] = [float(merkezler[kumeyeri[i]][0]) + float(X[i][0]),
                                          float(merkezler[kumeyeri[i]][1]) + float(X[i][1]),
                                          int(merkezler[kumeyeri[i]][2]) + 1]
            for i in range(len(merkezler)):
                merkezler[i] = [float(merkezler[i][0]) / float(merkezler[i][2]), (merkezler[i][1] / merkezler[i][2]),
                                int(merkezler[i][2])]
            kumeyeriyeni = []
            for i in range(len(X)):
                kumeyeriyeni.append(0)
            for i in range(len(X)):
                deger = [0, 0]
                deger = [math.sqrt(math.pow(abs(float(X[i][0]) - float(merkezler[0][0])), 2) + math.pow(
                    abs(float(X[i][1]) - float(merkezler[0][1])), 2)), 0]
                if (deger[0] > math.sqrt(math.pow(abs(float(X[i][0]) - float(merkezler[1][0])), 2) + math.pow(
                        abs(float(X[i][1]) - float(merkezler[1][1])), 2))):
                    deger = [math.sqrt(math.pow(abs(float(X[i][0]) - float(merkezler[1][0])), 2) + math.pow(
                        abs(float(X[i][1]) - float(merkezler[1][1])), 2)), 1]
                kumeyeriyeni[i] = deger[1]
            if (kumeyeriyeni == kumeyeri):
                devammi = False
            else:
                kumeyeri = kumeyeriyeni
        colors = ["g.", "r.", "b.", "y.", "c.", "m."]
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], colors[kumeyeri[i]], markersize=10)
        for i in range(len(merkezler)):
            plt.scatter(int(merkezler[i][0]), int(merkezler[i][1]), marker="x", s=70, linewidths=3, zorder=10,
                        c="black")
        plt.savefig('./sonuclar/knn/Knn.png')
        plt.show()
        w, h = self.KNN_sonuc_GV.width() - 5, self.KNN_sonuc_GV.height() - 5
        self.KNN_sonuc_GV.setScene(self.show_image('./sonuclar/knn/Knn.png', w, h))
        self.kumeye = kumeyeri

    def knn_veri_ekle_kumele(self):
        ekle = [int(self.KNN_X_lineEd.text()), int(self.KNN_Y_lineEd.text())]
        X = self.Knn
        merkezler = []
        kumeyeri = self.kumeye
        colors = ["g.", "r.", "b.", "y.", "c.", "m."]
        plt.plot(ekle[0], ekle[1], colors[5], markersize=20)
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], colors[kumeyeri[i]], markersize=10)
        for i in range(len(merkezler)):
            plt.scatter(int(merkezler[i][0]), int(merkezler[i][1]), marker="x", s=70, linewidths=3, zorder=10,
                        c="black")
        plt.show()
        degerler = []
        for i in range(len(X)):
            degerler.append(
                math.sqrt(math.pow(abs(float(X[i][0]) - ekle[0]), 2) + math.pow(abs(float(X[i][1]) - ekle[0]), 2)))
        bak = [[max(degerler), 0], [max(degerler), 0], [max(degerler), 0], [max(degerler), 0], [max(degerler), 0]]
        for i in range(len(degerler)):
            for j in range(len(bak)):
                if (degerler[i] < bak[j][0]):
                    if (degerler[i] != bak[j][0]):
                        bak[j] = [degerler[i], i]
                        break
        sifir = 0
        bir = 0
        for a in range(len(bak)):
            if (kumeyeri[bak[a][1]] == 1):
                bir = bir + 1
            else:
                sifir = sifir + 1
        renk = 0
        if (bir > sifir):
            renk = 1
        plt.plot(ekle[0], ekle[1], colors[renk], markersize=20)
        
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], colors[kumeyeri[i]], markersize=10)
        for i in range(len(merkezler)):
            plt.scatter(int(merkezler[i][0]), int(merkezler[i][1]), marker="x", s=70, linewidths=3, zorder=10,
                        c="black")
        plt.savefig('./sonuclar/knn/Knn-1.png')
        plt.show()
        w, h = self.KNN_sonuc_GV.width() - 5, self.KNN_sonuc_GV.height() - 5
        self.KNN_sonuc_GV.setScene(self.show_image('./sonuclar/knn/Knn-1.png', w, h))

# ********************************************* KNN TAB END *********************************************************


# ********************************** K - MEANS TAB BEGINNING *********************************************************

    colors = ["g.","r.","b.","y.", " c."]
    kumesayisi=3
    kumeyeri=[]
    XKmeans=[]

    def veriYukleKmeans(self):
        f = open('./veriler/veri.txt')
        
        X=[]
        for i,row in enumerate(f.readlines()):
            currentline = row.split(",")   
            temp=[]
            for column_value in currentline:
                temp.append(column_value)
            X.append(temp)
        
        
        for i in range(len(X)): 
            self.kumeyeri.append(int(X[i][0])%self.kumesayisi)
            
        self.XKmeans=X
        self.Kmeans_tablo.setColumnCount(2)
        self.Kmeans_tablo.setRowCount(len(self.XKmeans))  ##set number of rows
        for rowNumber, row in enumerate(self.XKmeans):
            self.Kmeans_tablo.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            self.Kmeans_tablo.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))

    def kumelemeKmeans(self):
        X=self.XKmeans
        devammi=True
        while(devammi):
            merkezler=[]
            for i in range(self.kumesayisi):
                merkezler.append([0,0,0])
            for i in range(len(X)):
                merkezler[self.kumeyeri[i]]=[float(merkezler[self.kumeyeri[i]][0])+float(X[i][0]),float(merkezler[self.kumeyeri[i]][1])+float(X[i][1]),int(merkezler[self.kumeyeri[i]][2])+1]
            for i in range(len(merkezler)):
                merkezler[i]=[float(merkezler[i][0])/float(merkezler[i][2]),(merkezler[i][1]/merkezler[i][2]),int(merkezler[i][2])]
            
            kumeyeriyeni=[]
            
            print merkezler
            for i in range(len(X)): 
                kumeyeriyeni.append(0)
            for i in range(len(X)):
                deger=[0,0]
                deger=[ math.sqrt(math.pow(abs(float(X[i][0])- float(merkezler[0][0])),2)+math.pow(abs(float(X[i][1])- float(merkezler[0][1])),2)) , 0]
                if(deger[0]> math.sqrt(math.pow(abs(float(X[i][0])- float(merkezler[1][0])),2)+math.pow(abs(float(X[i][1])- float(merkezler[1][1])),2))):
                    deger=[math.sqrt(math.pow(abs(float(X[i][0])- float(merkezler[1][0])),2)+math.pow(abs(float(X[i][1])- float(merkezler[1][1])),2)),1]
                if(deger[0]> math.sqrt(math.pow(abs(float(X[i][0])- float(merkezler[2][0])),2)+math.pow(abs(float(X[i][1])- float(merkezler[2][1])),2))):
                    deger=[math.sqrt(math.pow(abs(float(X[i][0])- float(merkezler[2][0])),2)+math.pow(abs(float(X[i][1])- float(merkezler[2][1])),2)),2]
                kumeyeriyeni[i]=deger[1]
           
            if(kumeyeriyeni==self.kumeyeri):
            
                devammi=False
                
            else:
    
                self.kumeyeri=kumeyeriyeni
       
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], self.colors[self.kumeyeri[i]], markersize = 10)
        for i in range(len(merkezler)):
            plt.scatter(int(merkezler[i][0]),int(merkezler[i][1]), marker = "x", s=70, linewidths = 3, zorder = 10,c="Black")
      
        plt.savefig('./sonuclar/kmeans/Kmeans.png')        
        plt.show()
         
        w, h = self.Kmeans_Kumele_GV.width() - 5, self.Kmeans_Kumele_GV.height() - 5
        self.Kmeans_Kumele_GV.setScene(self.show_image('./sonuclar/kmeans/Kmeans.png', w, h))
    


# ********************************** K - MEANS TAB END *********************************************************



# ********************************** Navie Bayes TAB BEGIGINING *********************************************************
    @QtCore.pyqtSignature("bool")
    def on_NavieBay_btn_clicked(self):   
        kelime=str(self.lineEdit.text())
        def aranacak(kume, kelime, index):
            counter=0
            for i in range(len(kume)):
                if kume[i][index]==kelime:
                    counter+=1
            return counter
    
        def aranacak_2(kume, kelime):
            counter=0
            for i in range(len(kume)):
                if kume[i]==kelime:
                    counter+=1
            return counter
    
        def kelimeler(kume):
            kelimeler=[]
            silinecek="!@#$.?,"
            for i in range(len(kume)):
                cumle=kume[i][0]
                for char in silinecek:
                    cumle=cumle.replace(char,"")
                parca=cumle.split(' ')
                for c in parca:
                    if aranacak_2(kelimeler, c)==0:
                        kelimeler.append(c)
            return kelimeler
    
        def arama(kume,kumeci,kelime):
            counter=0
            for i in range(len(kume)):
                if kume[i][1]==kumeci and kume[i][0].count(kelime)>0:
                    counter+=kume[i][0].count(kelime)
            return counter
            
        data=[["top, futbol, kondisyon, antrenman.","spor"],
            ["saha futbol fitness voleybol basketbol.","spor"],
            ["penalti, ofsayt,sut,tac, masa tenisi","spor"],
            ["ceza sahasi ,kale, top.","spor"],
            ["enflasyon, deflasyon, komisyon, sermaye, indeks","ekonomi"],
            ["lira, kar, zarar, altin, faiz, hisse","ekonomi"],
            ["bonus , piyasa, euro, tl, para, hesap","ekonomi"],
            ["finans, dolar, gelir","ekonomi"]]

        countspor=aranacak(data,"spor",1)
        countekonom=aranacak(data,"ekonomi",1)
        print("ekonomi adet:"+str(countekonom)+" spor Adet:"+str(countspor))
        sporagirlik=float(countspor)/(float(countspor)+float(countekonom))
        ekonomagirlik=float(countekonom)/(float(countspor)+float(countekonom))
        print("spor :"+str(sporagirlik)+" ekonomi :"+str(ekonomagirlik))
        kelimeci=kelimeler(data)
        print(kelimeler(data))
        sportoplam=0
        spordeger=[]
        for i in kelimeci:
            sportoplam+=(arama(data,"spor",i)+1)

        for i in range(len(kelimeci)):
            deger=float(arama(data,"spor",kelimeci[i])+1)/float(sportoplam)
            spordeger.append(deger)
            print(str(kelimeci[i])+" icin "+str(deger))
    
        ekonomtoplam=0
        ekonomdeger=[]
        for i in kelimeci:
            ekonomtoplam+=(arama(data,"ekonomi",i)+1)
        for i in range(len(kelimeci)):
            deger=float(arama(data,"ekonomi",kelimeci[i])+1)/float(ekonomtoplam)
            ekonomdeger.append(deger)
            print(str(kelimeci[i])+" icin "+str(deger))
        c_kelime=kelime.split(" ")
        print(c_kelime)
        sporcarpim=1
        for i in c_kelime:
            for x in range(len(kelimeci)):
                if kelimeci[x]==i:
                    sporcarpim*=spordeger[x]
        ekonomcarpim=1
        for i in c_kelime:
            for x in range(len(kelimeci)):
                if kelimeci[x]==i:
                    ekonomcarpim*=ekonomdeger[x]
        sporsonuc=sporcarpim*sporagirlik
        ekonomsonuc=ekonomcarpim*ekonomagirlik
        print("spor cumle oran:"+str(sporcarpim)+" Oran:"+str(sporsonuc))
        print("ekonomi cumle oran:"+str(ekonomcarpim)+" Oran:"+str(ekonomsonuc))
        
        if sporsonuc<ekonomsonuc:
            print("Kelime ekonomi")
            self.label_14.setText("Kelime Ekonomi")
        if sporsonuc>ekonomsonuc:
            print("Kelime spor")
            self.label_14.setText("Kelime spor")
            
        if sporcarpim==1 and ekonomcarpim==1:
            self.label_14.setText("Kelime yok")


# **********************************    Navie Bayes TAB END     *********************************************************





# ********************************** NORMALiZASTON TAB BEGIGINING **************************************************


    def dosyaYuklenormalize(self):
        f = open('./veriler/diabetes.data')
        X=[]

        for i,row in enumerate(f.readlines()):   
            currentline = row.split(",")   
            temp=[]

            for column_value in currentline:
                temp.append(column_value)
            X.append(temp)

        X=np.array(X)

        self.X=X[:,:8]
        self.y=X[:,8]
        self.veriYukle(self.X,self.y,self.Norm_tablo_veriYukle)


    
    def veriYukle(self,X,y,tablonormalize):
        num_rows=len(X)
        tablonormalize.clear()    
        tablonormalize.setColumnCount(8)
        tablonormalize.setRowCount(num_rows) ##set number of rows

        for rowNumber,row in enumerate(X):

            #row[1].encode("utf-8")
            tablonormalize.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tablonormalize.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tablonormalize.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tablonormalize.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            tablonormalize.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4])))
            tablonormalize.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))
            tablonormalize.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(row[6])))
            tablonormalize.setItem(rowNumber, 7, QtGui.QTableWidgetItem(str(row[7])))

        for rowNumber,row in enumerate(y):

            tablonormalize.setItem(rowNumber, 8, QtGui.QTableWidgetItem(str(row)))


    def normalize_MinMax(self):
        for s in range(0,8):
            first_column=self.X[:,s]
            max_value=float(max(first_column))
            min_value=float(min(first_column))
            print "max value:",max_value," min value:",min_value
            num_rows=len(self.X)
            for i,value in enumerate(first_column):
               normalize_value=((float(value)-min_value)/(max_value-min_value))
               first_column[i]=round(normalize_value,2)   
            self.Norm_tablo_MinMax.setColumnCount(8)
            self.Norm_tablo_MinMax.setRowCount(num_rows)
                    
            for rowNumber,row in enumerate(first_column):
                self.Norm_tablo_MinMax.setItem(rowNumber, s, QtGui.QTableWidgetItem(str(row)))
                
    def normalize_Z_Score(self):
        for s in range(0,8):
            colm= np.array(self.X[:,s]).astype(np.float)
            ui=np.mean(colm)
            ai=np.std(colm)
            print"Aritmetik ortalama:",ui,"  standart sapma:",ai
            num_rows=len(self.X)
            for i,value in enumerate(colm):
                normalize_zscor=float(value)-ui/ai
                colm[i]=float(round(normalize_zscor,3))
            self.Norm_tablo_Zscore.setColumnCount(8)
            self.Norm_tablo_Zscore.setRowCount(num_rows)                
            for rowNumber,row in enumerate(colm):
                self.Norm_tablo_Zscore.setItem(rowNumber, s, QtGui.QTableWidgetItem(str(row)))
                            
    def normalizeMedian(self):
        for s in range(0,8):
            column=np.array(self.X[:,s]).astype(np.float)
            med=np.median(column)
            print "Medyan: ",med
            num_rows=len(self.X)
            for i,value in enumerate(column):
                normalize_medyan=float(value)/med
                column[i]=float(round(normalize_medyan,3))
            self.Norm_tablo_Median.setColumnCount(8)
            self.Norm_tablo_Median.setRowCount(num_rows)
                
            for rowNumber,row in enumerate(column):
                self.Norm_tablo_Median.setItem(rowNumber, s, QtGui.QTableWidgetItem(str(row)))



# ********************************** NORMALiZASTON TAB END **************************************************


# ********************************** RANDOM FOREST TAB BEGIGINING **************************************************

    def  RandomForest(self):
        
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(self.X_train, self.y_train)
        results=clf.predict(self.X_test)
        self.label_17.setText(str(round(accuracy_score(self.y_test, results)*100,2)))


# ********************************** RANDOM FOREST TAB END *********************************************************


# ***************************************** TRAIN - TEST TAB  *****************************************************

    def tum_veri_ekle(self):
        f = open('./veriler/veri.data')
        X = []
        self.Xveri = []
        self.Yveri = []

        for i, row in enumerate(f.readlines()):
            currentline = row.split(",")
            temp = []
            for column_value in currentline:
                temp.append(column_value)
            X.append(temp)
            self.Xveri.append(temp[:8])
            self.Yveri.append(temp[8])
        self.TT_tummu_Table.clear()
        self.TT_tummu_Table.setColumnCount(9)
        self.TT_tummu_Table.setRowCount(len(X))  ##set number of rows

        for rowNumber, row in enumerate(X):

            self.TT_tummu_Table.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            self.TT_tummu_Table.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.TT_tummu_Table.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            self.TT_tummu_Table.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            self.TT_tummu_Table.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4])))
            self.TT_tummu_Table.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))
            self.TT_tummu_Table.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(row[6])))
            self.TT_tummu_Table.setItem(rowNumber, 7, QtGui.QTableWidgetItem(str(row[7])))
            self.TT_tummu_Table.setItem(rowNumber, 8, QtGui.QTableWidgetItem(str(row[8])))

    def bol_veri_tt(self):
        yuzde=float(float(self.TT_comboBox.currentText())/100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.Xveri, self.Yveri, test_size=yuzde, random_state=42)

        self.TT_train_Table.clear()
        self.TT_train_Table.setColumnCount(8)
        self.TT_train_Table.setRowCount(len(self.X_train))  ##set number of rows TT_train_Table

        for rowNumber, row in enumerate(self.X_train):
            self.TT_train_Table.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            self.TT_train_Table.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.TT_train_Table.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            self.TT_train_Table.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            self.TT_train_Table.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4])))
            self.TT_train_Table.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))
            self.TT_train_Table.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(row[6])))
            self.TT_train_Table.setItem(rowNumber, 7, QtGui.QTableWidgetItem(str(row[7])))
        self.TT_test_Table.clear()
        self.TT_test_Table.setColumnCount(8)
        self.TT_test_Table.setRowCount(len(self.X_test))  ##set number of rows

        for rowNumber, row in enumerate(self.X_test):
            self.TT_test_Table.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            self.TT_test_Table.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.TT_test_Table.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            self.TT_test_Table.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            self.TT_test_Table.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4])))
            self.TT_test_Table.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))
            self.TT_test_Table.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(row[6])))
            self.TT_test_Table.setItem(rowNumber, 7, QtGui.QTableWidgetItem(str(row[7])))

# ***************************************** TRAIN - TEST TAB END *****************************************************

# ***************************************** PARKINSON TAB  *****************************************************

    def butunbtn(self):
        self.X_trainp, self.X_tesptp, self.y_trainp, self.y_testp =train_test_split(self.Topalamx,self.Topalamy , test_size=0.30, random_state=42)

    def dstbtn(self):
        self.X_trainp, self.X_tesptp, self.y_trainp, self.y_testp =self.DSTx, self.DSTxn,self.DSTy,self.DSTyn

    def stcpbtn(self):
        self.X_trainp, self.X_tesptp, self.y_trainp, self.y_testp =self.STCPx, self.STCPxn,self.STCPy,self.STCPyn

    def RandomFogren(self):
        clf = RandomForestClassifier(max_depth=None, random_state=0)
        clf.fit(self.X_trainp,self.y_trainp)
        results=clf.predict(self.X_tesptp)        
        self.label_64.setText("Basari Yuzdesi :"+str(float("{0:.2f}".format(accuracy_score(self.y_testp,results)*100))))

    def sstbtn(self):
        self.X_trainp, self.X_tesptp, self.y_trainp, self.y_testp=self.SSTx, self.SSTxn,self.SSTy,self.SSTyn

    def parkisyonveri(self):
        pathparkinson="./hw_dataset/parkinson/"     
        pathcontrol="./hw_dataset/control/"     
        pathparkinsonnew="./new_dataset/parkinson/"   
        SST=[]   #0
        DST=[]   #1
        STCP=[]  #2
        
        SST_train=[]   #0
        DST_train=[]   #1
        STCP_train=[]  #2
        
        SST_test=[]   #0
        DST_test=[]   #1
        STCP_test=[]  #2
        
        dosyalar=os.listdir(pathparkinson)
        for dosya in dosyalar:
            f = open(pathparkinson+dosya)        
            for i,row in enumerate(f.readlines()):
                currentline = row.split(";")   
                temp=[]
                for column_value in currentline:
                    temp.append(column_value)
                    
                if(int(temp[len(temp)-1])==0):
                    temp.remove(temp[len(temp)-1])
                    SST_train.append(temp[:6])
                    SST_test.append(1) 
                    temp.append(1)            
                    SST.append(temp)           
                elif(int(temp[len(temp)-1])==1):
                    temp.remove(temp[len(temp)-1])
                    DST_train.append(temp[:6])
                    DST_test.append(1) 
                    temp.append(1) 
                    DST.append(temp)
                elif(int(temp[len(temp)-1])==2):
                    temp.remove(temp[len(temp)-1])
                    STCP_train.append(temp[:6])
                    STCP_test.append(1) 
                    temp.append(1)   
                    STCP.append(temp)

        dosyalar=os.listdir(pathcontrol)
        for dosya in dosyalar:
            f = open(pathcontrol+dosya)        
            for i,row in enumerate(f.readlines()):
                currentline = row.split(";")   
                temp=[]
                for column_value in currentline:
                    temp.append(column_value)
                if(int(temp[len(temp)-1])==0):
                    temp.remove(temp[len(temp)-1])
                    SST_train.append(temp[:6])
                    SST_test.append(0) 
                    temp.append(0)            
                    SST.append(temp)           
                elif(int(temp[len(temp)-1])==1):
                    temp.remove(temp[len(temp)-1])
                    DST_train.append(temp[:6])
                    DST_test.append(0)  
                    temp.append(0)  
                    DST.append(temp)
                elif(int(temp[len(temp)-1])==2):
                    temp.remove(temp[len(temp)-1])
                    STCP_train.append(temp[:6])
                    STCP_test.append(0) 
                    temp.append(0)   
                    STCP.append(temp)
        SST_new=[]   #0
        DST_new=[]   #1
        STCP_new=[]  #2
        
        SST_train_new=[]   #0
        DST_train_new=[]   #1
        STCP_train_new=[]  #2
        
        SST_test_new=[]   #0
        DST_test_new=[]   #1
        STCP_test_new=[]  #2
        dosyalar=os.listdir(pathparkinsonnew)
        for dosya in dosyalar:
            f = open(pathparkinsonnew+dosya)        
            for i,row in enumerate(f.readlines()):
                currentline = row.split(";")   
                temp=[]
                for column_value in currentline:
                    temp.append(column_value)
                if(int(temp[len(temp)-1])==0):
                    temp.remove(temp[len(temp)-1])
                    SST_train_new.append(temp[:6])
                    SST_test_new.append(1) 
                    temp.append(1)            
                    SST_new.append(temp)           
                elif(int(temp[len(temp)-1])==1):
                    temp.remove(temp[len(temp)-1])
                    DST_train_new.append(temp[:6])
                    DST_test_new.append(1) 
                    temp.append(1) 
                    DST_new.append(temp)
                elif(int(temp[len(temp)-1])==2):
                    temp.remove(temp[len(temp)-1])
                    STCP_train_new.append(temp[:6])
                    STCP_test_new.append(1) 
                    temp.append(1)   
                    STCP_new.append(temp)
            
            
            self.SSTx=SST_train
            self.SSTy=SST_test
            self.SSTxn=SST_train_new
            self.SSTyn=SST_test_new
            
            self.DSTx=DST_train
            self.DSTy=DST_test
            self.DSTxn=DST_train_new
            self.DSTyn=DST_test_new
            
            self.STCPx=STCP_train
            self.STCPy=STCP_test
            self.STCPxn=STCP_train_new
            self.STCPyn=STCP_test_new
            
            
            toplam_train=[]
            toplam_test=[]
            
            toplam_train.extend(SST_train)
            toplam_test.extend(STCP_test)
            toplam_train.extend(STCP_train_new)
            toplam_test.extend(SST_test_new)
            
            toplam_train.extend(DST_train)
            toplam_test.extend(DST_test)
            toplam_train.extend(DST_train_new)
            toplam_test.extend(DST_test_new)
            
            toplam_train.extend(STCP_train)
            toplam_test.extend(SST_test)
            toplam_train.extend(SST_train_new)
            toplam_test.extend(STCP_test_new)
            
            self.Topalamx=toplam_train
            self.Topalamy=toplam_test
        self.tableWidget_1_9.clear()
        self.tableWidget_1_9.setColumnCount(7)
        self.tableWidget_1_9.setRowCount(len(SST_train)) ##set number of rows
        for rowNumber,row in enumerate(SST_train):
            self.tableWidget_1_9.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.tableWidget_1_9.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.tableWidget_1_9.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2]))) 
            self.tableWidget_1_9.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3]))) 
            self.tableWidget_1_9.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            self.tableWidget_1_9.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))  
            self.tableWidget_1_9.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(SST_test[rowNumber])))
            
            
        self.tableWidget_1_10.clear()
        self.tableWidget_1_10.setColumnCount(7)
        self.tableWidget_1_10.setRowCount(len(DST_train)) ##set number of rows
        for rowNumber,row in enumerate(DST_train):
            self.tableWidget_1_10.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.tableWidget_1_10.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.tableWidget_1_10.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2]))) 
            self.tableWidget_1_10.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3]))) 
            self.tableWidget_1_10.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            self.tableWidget_1_10.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))  
            self.tableWidget_1_10.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(DST_test[rowNumber])))
        
        self.tableWidget_1_14.clear()
        self.tableWidget_1_14.setColumnCount(7)
        self.tableWidget_1_14.setRowCount(len(STCP_train_new)) ##set number of rows
        for rowNumber,row in enumerate(STCP_train_new):
            self.tableWidget_1_14.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.tableWidget_1_14.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.tableWidget_1_14.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2]))) 
            self.tableWidget_1_14.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3]))) 
            self.tableWidget_1_14.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            self.tableWidget_1_14.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))  
            self.tableWidget_1_14.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(STCP_test_new[rowNumber])))
            
            
            
        self.tableWidget_1_12.clear()
        self.tableWidget_1_12.setColumnCount(7)
        self.tableWidget_1_12.setRowCount(len(SST_train_new)) ##set number of rows
        for rowNumber,row in enumerate(SST_train_new):
            self.tableWidget_1_12.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.tableWidget_1_12.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.tableWidget_1_12.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2]))) 
            self.tableWidget_1_12.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3]))) 
            self.tableWidget_1_12.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            self.tableWidget_1_12.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))  
            self.tableWidget_1_12.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(SST_test_new[rowNumber])))
            
            
        self.tableWidget_1_13.clear()
        self.tableWidget_1_13.setColumnCount(7)
        self.tableWidget_1_13.setRowCount(len(DST_train_new)) ##set number of rows
        for rowNumber,row in enumerate(DST_train_new):
            self.tableWidget_1_13.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.tableWidget_1_13.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.tableWidget_1_13.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2]))) 
            self.tableWidget_1_13.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3]))) 
            self.tableWidget_1_13.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            self.tableWidget_1_13.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))  
            self.tableWidget_1_13.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(DST_test_new[rowNumber])))
        
        self.tableWidget_1_11.clear()
        self.tableWidget_1_11.setColumnCount(7)
        self.tableWidget_1_11.setRowCount(len(STCP_train)) ##set number of rows
        for rowNumber,row in enumerate(STCP_train):
            self.tableWidget_1_11.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0]))) 
            self.tableWidget_1_11.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            self.tableWidget_1_11.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2]))) 
            self.tableWidget_1_11.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3]))) 
            self.tableWidget_1_11.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            self.tableWidget_1_11.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))  
            self.tableWidget_1_11.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(STCP_test[rowNumber])))
        print "okuma tamamlandi"
            


# ***************************************** PARKINSON TAB END  *****************************************************
