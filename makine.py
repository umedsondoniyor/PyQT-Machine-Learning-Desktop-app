# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'makine.ui'
#
# Created: Sat Jan 13 18:50:11 2018
#      by: PyQt4 UI code generator 4.11
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(970, 571)
        Dialog.setStyleSheet(_fromUtf8(""))
        self.tabWidget = QtGui.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 971, 571))
        self.tabWidget.setAcceptDrops(False)
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet(_fromUtf8("color: rgb(0, 0, 0);\n"
"background-color: rgb(53, 255, 178);"))
        self.tabWidget.setTabPosition(QtGui.QTabWidget.South)
        self.tabWidget.setTabShape(QtGui.QTabWidget.Triangular)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.RR_veriYukle1_btn = QtGui.QPushButton(self.tab)
        self.RR_veriYukle1_btn.setGeometry(QtCore.QRect(20, 10, 81, 29))
        self.RR_veriYukle1_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.RR_veriYukle1_btn.setObjectName(_fromUtf8("RR_veriYukle1_btn"))
        self.RR_RveriOlustur1_btn = QtGui.QPushButton(self.tab)
        self.RR_RveriOlustur1_btn.setGeometry(QtCore.QRect(20, 50, 111, 29))
        self.RR_RveriOlustur1_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.RR_RveriOlustur1_btn.setObjectName(_fromUtf8("RR_RveriOlustur1_btn"))
        self.RR_Varalik1_1_lineEd = QtGui.QLineEdit(self.tab)
        self.RR_Varalik1_1_lineEd.setGeometry(QtCore.QRect(330, 10, 41, 29))
        self.RR_Varalik1_1_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(162, 255, 217);"))
        self.RR_Varalik1_1_lineEd.setObjectName(_fromUtf8("RR_Varalik1_1_lineEd"))
        self.RR_Varalik1_2_lineEd = QtGui.QLineEdit(self.tab)
        self.RR_Varalik1_2_lineEd.setGeometry(QtCore.QRect(380, 10, 41, 29))
        self.RR_Varalik1_2_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(162, 255, 217);"))
        self.RR_Varalik1_2_lineEd.setObjectName(_fromUtf8("RR_Varalik1_2_lineEd"))
        self.label = QtGui.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(110, 20, 71, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.RR_miktar1_lineEd = QtGui.QLineEdit(self.tab)
        self.RR_miktar1_lineEd.setGeometry(QtCore.QRect(190, 10, 61, 29))
        self.RR_miktar1_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(162, 255, 217);"))
        self.RR_miktar1_lineEd.setObjectName(_fromUtf8("RR_miktar1_lineEd"))
        self.label_2 = QtGui.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(260, 20, 71, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.RR_miktar2_lineEd = QtGui.QLineEdit(self.tab)
        self.RR_miktar2_lineEd.setGeometry(QtCore.QRect(710, 10, 61, 29))
        self.RR_miktar2_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(162, 255, 217);"))
        self.RR_miktar2_lineEd.setObjectName(_fromUtf8("RR_miktar2_lineEd"))
        self.RR_RveriOlustur2_btn = QtGui.QPushButton(self.tab)
        self.RR_RveriOlustur2_btn.setGeometry(QtCore.QRect(540, 50, 111, 29))
        self.RR_RveriOlustur2_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.RR_RveriOlustur2_btn.setObjectName(_fromUtf8("RR_RveriOlustur2_btn"))
        self.label_3 = QtGui.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(630, 20, 71, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.RR_veriYukle2_btn = QtGui.QPushButton(self.tab)
        self.RR_veriYukle2_btn.setGeometry(QtCore.QRect(540, 10, 81, 29))
        self.RR_veriYukle2_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.RR_veriYukle2_btn.setObjectName(_fromUtf8("RR_veriYukle2_btn"))
        self.label_4 = QtGui.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(780, 20, 71, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.RR_Varalik2_1_lineEd = QtGui.QLineEdit(self.tab)
        self.RR_Varalik2_1_lineEd.setGeometry(QtCore.QRect(850, 10, 41, 29))
        self.RR_Varalik2_1_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(162, 255, 217);"))
        self.RR_Varalik2_1_lineEd.setObjectName(_fromUtf8("RR_Varalik2_1_lineEd"))
        self.RR_Varalik2_2_lineEd = QtGui.QLineEdit(self.tab)
        self.RR_Varalik2_2_lineEd.setGeometry(QtCore.QRect(900, 10, 41, 29))
        self.RR_Varalik2_2_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(162, 255, 217);"))
        self.RR_Varalik2_2_lineEd.setObjectName(_fromUtf8("RR_Varalik2_2_lineEd"))
        self.RR_1_tabelW = QtGui.QTableWidget(self.tab)
        self.RR_1_tabelW.setGeometry(QtCore.QRect(145, 50, 281, 191))
        self.RR_1_tabelW.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.RR_1_tabelW.setObjectName(_fromUtf8("RR_1_tabelW"))
        self.RR_2_tabelW = QtGui.QTableWidget(self.tab)
        self.RR_2_tabelW.setGeometry(QtCore.QRect(665, 50, 281, 191))
        self.RR_2_tabelW.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.RR_2_tabelW.setObjectName(_fromUtf8("RR_2_tabelW"))
        self.RR_TUM_VERILER_GV = QtGui.QGraphicsView(self.tab)
        self.RR_TUM_VERILER_GV.setGeometry(QtCore.QRect(670, 290, 281, 231))
        self.RR_TUM_VERILER_GV.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.RR_TUM_VERILER_GV.setObjectName(_fromUtf8("RR_TUM_VERILER_GV"))
        self.RR_RUS_GV = QtGui.QGraphicsView(self.tab)
        self.RR_RUS_GV.setGeometry(QtCore.QRect(10, 290, 281, 231))
        self.RR_RUS_GV.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.RR_RUS_GV.setObjectName(_fromUtf8("RR_RUS_GV"))
        self.RR_ROS_GV = QtGui.QGraphicsView(self.tab)
        self.RR_ROS_GV.setGeometry(QtCore.QRect(300, 290, 271, 231))
        self.RR_ROS_GV.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.RR_ROS_GV.setObjectName(_fromUtf8("RR_ROS_GV"))
        self.RR_RUS_btn = QtGui.QPushButton(self.tab)
        self.RR_RUS_btn.setGeometry(QtCore.QRect(10, 250, 81, 29))
        self.RR_RUS_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.RR_RUS_btn.setObjectName(_fromUtf8("RR_RUS_btn"))
        self.RR_ROS_btn = QtGui.QPushButton(self.tab)
        self.RR_ROS_btn.setGeometry(QtCore.QRect(300, 250, 81, 29))
        self.RR_ROS_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.RR_ROS_btn.setObjectName(_fromUtf8("RR_ROS_btn"))
        self.label_5 = QtGui.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(670, 260, 91, 17))
        self.label_5.setStyleSheet(_fromUtf8("background-color: rgb(255, 222, 55);"))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.KNN_1_tabWidget = QtGui.QTableWidget(self.tab_2)
        self.KNN_1_tabWidget.setGeometry(QtCore.QRect(10, 50, 256, 461))
        self.KNN_1_tabWidget.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.KNN_1_tabWidget.setObjectName(_fromUtf8("KNN_1_tabWidget"))
        self.KNN_veriYukle_btn = QtGui.QPushButton(self.tab_2)
        self.KNN_veriYukle_btn.setGeometry(QtCore.QRect(10, 10, 81, 29))
        self.KNN_veriYukle_btn.setStyleSheet(_fromUtf8("background-color: rgb(180, 255, 130);"))
        self.KNN_veriYukle_btn.setObjectName(_fromUtf8("KNN_veriYukle_btn"))
        self.KNN_veriekle_GV = QtGui.QGraphicsView(self.tab_2)
        self.KNN_veriekle_GV.setGeometry(QtCore.QRect(275, 170, 331, 341))
        self.KNN_veriekle_GV.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.KNN_veriekle_GV.setObjectName(_fromUtf8("KNN_veriekle_GV"))
        self.KNN_sonuc_GV = QtGui.QGraphicsView(self.tab_2)
        self.KNN_sonuc_GV.setGeometry(QtCore.QRect(610, 170, 351, 341))
        self.KNN_sonuc_GV.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.KNN_sonuc_GV.setObjectName(_fromUtf8("KNN_sonuc_GV"))
        self.KNN_YeniveriYukle_btn = QtGui.QPushButton(self.tab_2)
        self.KNN_YeniveriYukle_btn.setGeometry(QtCore.QRect(460, 40, 121, 29))
        self.KNN_YeniveriYukle_btn.setStyleSheet(_fromUtf8("background-color: rgb(180, 255, 130);"))
        self.KNN_YeniveriYukle_btn.setObjectName(_fromUtf8("KNN_YeniveriYukle_btn"))
        self.KNN_X_lineEd = QtGui.QLineEdit(self.tab_2)
        self.KNN_X_lineEd.setGeometry(QtCore.QRect(610, 40, 41, 29))
        self.KNN_X_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(138, 255, 246);"))
        self.KNN_X_lineEd.setObjectName(_fromUtf8("KNN_X_lineEd"))
        self.KNN_Y_lineEd = QtGui.QLineEdit(self.tab_2)
        self.KNN_Y_lineEd.setGeometry(QtCore.QRect(690, 40, 41, 29))
        self.KNN_Y_lineEd.setStyleSheet(_fromUtf8("background-color: rgb(138, 255, 246);"))
        self.KNN_Y_lineEd.setObjectName(_fromUtf8("KNN_Y_lineEd"))
        self.KNN_Kumeleme_btn = QtGui.QPushButton(self.tab_2)
        self.KNN_Kumeleme_btn.setGeometry(QtCore.QRect(530, 98, 171, 51))
        self.KNN_Kumeleme_btn.setStyleSheet(_fromUtf8("background-color: rgb(180, 255, 130);"))
        self.KNN_Kumeleme_btn.setObjectName(_fromUtf8("KNN_Kumeleme_btn"))
        self.label_8 = QtGui.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(620, 20, 16, 17))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.label_9 = QtGui.QLabel(self.tab_2)
        self.label_9.setGeometry(QtCore.QRect(710, 20, 16, 17))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.Kmeans_tablo = QtGui.QTableWidget(self.tab_3)
        self.Kmeans_tablo.setGeometry(QtCore.QRect(10, 50, 256, 451))
        self.Kmeans_tablo.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.Kmeans_tablo.setObjectName(_fromUtf8("Kmeans_tablo"))
        self.Kmeans_veriYukle_btn = QtGui.QPushButton(self.tab_3)
        self.Kmeans_veriYukle_btn.setGeometry(QtCore.QRect(10, 10, 81, 29))
        self.Kmeans_veriYukle_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 144, 46);"))
        self.Kmeans_veriYukle_btn.setObjectName(_fromUtf8("Kmeans_veriYukle_btn"))
        self.Kmeans_Kumele_btn = QtGui.QPushButton(self.tab_3)
        self.Kmeans_Kumele_btn.setGeometry(QtCore.QRect(370, 10, 91, 31))
        self.Kmeans_Kumele_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 144, 46);"))
        self.Kmeans_Kumele_btn.setObjectName(_fromUtf8("Kmeans_Kumele_btn"))
        self.Kmeans_Kumele_GV = QtGui.QGraphicsView(self.tab_3)
        self.Kmeans_Kumele_GV.setGeometry(QtCore.QRect(370, 50, 561, 451))
        self.Kmeans_Kumele_GV.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.Kmeans_Kumele_GV.setObjectName(_fromUtf8("Kmeans_Kumele_GV"))
        self.tabWidget.addTab(self.tab_3, _fromUtf8(""))
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))
        self.lineEdit = QtGui.QLineEdit(self.tab_4)
        self.lineEdit.setGeometry(QtCore.QRect(240, 70, 441, 31))
        self.lineEdit.setStyleSheet(_fromUtf8("background-color: rgb(234, 254, 255);"))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.NavieBay_btn = QtGui.QPushButton(self.tab_4)
        self.NavieBay_btn.setGeometry(QtCore.QRect(370, 110, 181, 71))
        self.NavieBay_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 210, 94);"))
        self.NavieBay_btn.setObjectName(_fromUtf8("NavieBay_btn"))
        self.label_12 = QtGui.QLabel(self.tab_4)
        self.label_12.setGeometry(QtCore.QRect(420, 40, 101, 16))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.label_13 = QtGui.QLabel(self.tab_4)
        self.label_13.setGeometry(QtCore.QRect(170, 250, 121, 31))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.label_14 = QtGui.QLabel(self.tab_4)
        self.label_14.setGeometry(QtCore.QRect(380, 250, 121, 31))
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.tabWidget.addTab(self.tab_4, _fromUtf8(""))
        self.tab_7 = QtGui.QWidget()
        self.tab_7.setObjectName(_fromUtf8("tab_7"))
        self.Norm_tablo_veriYukle = QtGui.QTableWidget(self.tab_7)
        self.Norm_tablo_veriYukle.setGeometry(QtCore.QRect(10, 10, 361, 241))
        self.Norm_tablo_veriYukle.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.Norm_tablo_veriYukle.setObjectName(_fromUtf8("Norm_tablo_veriYukle"))
        self.Norm_btn_MinMax = QtGui.QPushButton(self.tab_7)
        self.Norm_btn_MinMax.setGeometry(QtCore.QRect(500, 80, 81, 81))
        self.Norm_btn_MinMax.setStyleSheet(_fromUtf8("background-color: rgb(0, 170, 255);"))
        self.Norm_btn_MinMax.setObjectName(_fromUtf8("Norm_btn_MinMax"))
        self.Norm_btn_veriYukle = QtGui.QPushButton(self.tab_7)
        self.Norm_btn_veriYukle.setGeometry(QtCore.QRect(380, 80, 81, 81))
        self.Norm_btn_veriYukle.setStyleSheet(_fromUtf8("background-color: rgb(0, 170, 255);"))
        self.Norm_btn_veriYukle.setObjectName(_fromUtf8("Norm_btn_veriYukle"))
        self.Norm_tablo_MinMax = QtGui.QTableWidget(self.tab_7)
        self.Norm_tablo_MinMax.setGeometry(QtCore.QRect(590, 10, 371, 241))
        self.Norm_tablo_MinMax.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.Norm_tablo_MinMax.setObjectName(_fromUtf8("Norm_tablo_MinMax"))
        self.Norm_btn_Zscore = QtGui.QPushButton(self.tab_7)
        self.Norm_btn_Zscore.setGeometry(QtCore.QRect(380, 350, 81, 81))
        self.Norm_btn_Zscore.setStyleSheet(_fromUtf8("background-color: rgb(0, 170, 255);"))
        self.Norm_btn_Zscore.setObjectName(_fromUtf8("Norm_btn_Zscore"))
        self.Norm_btn_Median = QtGui.QPushButton(self.tab_7)
        self.Norm_btn_Median.setGeometry(QtCore.QRect(500, 350, 81, 81))
        self.Norm_btn_Median.setStyleSheet(_fromUtf8("background-color: rgb(0, 170, 255);"))
        self.Norm_btn_Median.setObjectName(_fromUtf8("Norm_btn_Median"))
        self.Norm_tablo_Zscore = QtGui.QTableWidget(self.tab_7)
        self.Norm_tablo_Zscore.setGeometry(QtCore.QRect(10, 270, 361, 251))
        self.Norm_tablo_Zscore.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.Norm_tablo_Zscore.setObjectName(_fromUtf8("Norm_tablo_Zscore"))
        self.Norm_tablo_Median = QtGui.QTableWidget(self.tab_7)
        self.Norm_tablo_Median.setGeometry(QtCore.QRect(590, 270, 371, 251))
        self.Norm_tablo_Median.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.Norm_tablo_Median.setObjectName(_fromUtf8("Norm_tablo_Median"))
        self.tabWidget.addTab(self.tab_7, _fromUtf8(""))
        self.tab_8 = QtGui.QWidget()
        self.tab_8.setObjectName(_fromUtf8("tab_8"))
        self.TT_tummu_Table = QtGui.QTableWidget(self.tab_8)
        self.TT_tummu_Table.setGeometry(QtCore.QRect(10, 70, 291, 441))
        self.TT_tummu_Table.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.TT_tummu_Table.setObjectName(_fromUtf8("TT_tummu_Table"))
        self.TT_train_Table = QtGui.QTableWidget(self.tab_8)
        self.TT_train_Table.setGeometry(QtCore.QRect(330, 70, 256, 441))
        self.TT_train_Table.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.TT_train_Table.setObjectName(_fromUtf8("TT_train_Table"))
        self.TT_test_Table = QtGui.QTableWidget(self.tab_8)
        self.TT_test_Table.setGeometry(QtCore.QRect(680, 70, 256, 441))
        self.TT_test_Table.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.TT_test_Table.setObjectName(_fromUtf8("TT_test_Table"))
        self.label_6 = QtGui.QLabel(self.tab_8)
        self.label_6.setGeometry(QtCore.QRect(10, 50, 91, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_7 = QtGui.QLabel(self.tab_8)
        self.label_7.setGeometry(QtCore.QRect(330, 50, 91, 17))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.label_10 = QtGui.QLabel(self.tab_8)
        self.label_10.setGeometry(QtCore.QRect(680, 50, 81, 17))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.TT_upload_btn = QtGui.QPushButton(self.tab_8)
        self.TT_upload_btn.setGeometry(QtCore.QRect(10, 10, 89, 29))
        self.TT_upload_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 217, 65);"))
        self.TT_upload_btn.setObjectName(_fromUtf8("TT_upload_btn"))
        self.label_11 = QtGui.QLabel(self.tab_8)
        self.label_11.setGeometry(QtCore.QRect(330, 20, 91, 17))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.TT_comboBox = QtGui.QComboBox(self.tab_8)
        self.TT_comboBox.setGeometry(QtCore.QRect(420, 10, 71, 31))
        self.TT_comboBox.setStyleSheet(_fromUtf8("background-color: rgb(255, 252, 151);"))
        self.TT_comboBox.setObjectName(_fromUtf8("TT_comboBox"))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_comboBox.addItem(_fromUtf8(""))
        self.TT_uygula_btn = QtGui.QPushButton(self.tab_8)
        self.TT_uygula_btn.setGeometry(QtCore.QRect(500, 10, 101, 31))
        self.TT_uygula_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 217, 65);"))
        self.TT_uygula_btn.setObjectName(_fromUtf8("TT_uygula_btn"))
        self.RanForest_uygula_btn = QtGui.QPushButton(self.tab_8)
        self.RanForest_uygula_btn.setGeometry(QtCore.QRect(630, 10, 101, 31))
        self.RanForest_uygula_btn.setStyleSheet(_fromUtf8("background-color: rgb(255, 217, 65);"))
        self.RanForest_uygula_btn.setObjectName(_fromUtf8("RanForest_uygula_btn"))
        self.label_16 = QtGui.QLabel(self.tab_8)
        self.label_16.setGeometry(QtCore.QRect(740, 20, 91, 17))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.label_17 = QtGui.QLabel(self.tab_8)
        self.label_17.setGeometry(QtCore.QRect(820, 20, 141, 17))
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.tabWidget.addTab(self.tab_8, _fromUtf8(""))
        self.tab_9 = QtGui.QWidget()
        self.tab_9.setObjectName(_fromUtf8("tab_9"))
        self.label_61 = QtGui.QLabel(self.tab_9)
        self.label_61.setGeometry(QtCore.QRect(540, 0, 211, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_61.setFont(font)
        self.label_61.setObjectName(_fromUtf8("label_61"))
        self.label_59 = QtGui.QLabel(self.tab_9)
        self.label_59.setGeometry(QtCore.QRect(10, 140, 211, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_59.setFont(font)
        self.label_59.setObjectName(_fromUtf8("label_59"))
        self.tableWidget_1_14 = QtGui.QTableWidget(self.tab_9)
        self.tableWidget_1_14.setGeometry(QtCore.QRect(530, 300, 400, 111))
        self.tableWidget_1_14.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.tableWidget_1_14.setObjectName(_fromUtf8("tableWidget_1_14"))
        self.tableWidget_1_12 = QtGui.QTableWidget(self.tab_9)
        self.tableWidget_1_12.setGeometry(QtCore.QRect(530, 20, 400, 111))
        self.tableWidget_1_12.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.tableWidget_1_12.setObjectName(_fromUtf8("tableWidget_1_12"))
        self.label_58 = QtGui.QLabel(self.tab_9)
        self.label_58.setGeometry(QtCore.QRect(20, 0, 211, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_58.setFont(font)
        self.label_58.setObjectName(_fromUtf8("label_58"))
        self.label_62 = QtGui.QLabel(self.tab_9)
        self.label_62.setGeometry(QtCore.QRect(540, 140, 201, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_62.setFont(font)
        self.label_62.setObjectName(_fromUtf8("label_62"))
        self.tableWidget_1_10 = QtGui.QTableWidget(self.tab_9)
        self.tableWidget_1_10.setGeometry(QtCore.QRect(10, 160, 400, 111))
        self.tableWidget_1_10.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.tableWidget_1_10.setObjectName(_fromUtf8("tableWidget_1_10"))
        self.pushButton_6 = QtGui.QPushButton(self.tab_9)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 430, 111, 71))
        self.pushButton_6.setStyleSheet(_fromUtf8("background-color: rgb(164, 157, 255);"))
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_6"))
        self.tableWidget_1_13 = QtGui.QTableWidget(self.tab_9)
        self.tableWidget_1_13.setGeometry(QtCore.QRect(530, 160, 400, 111))
        self.tableWidget_1_13.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.tableWidget_1_13.setObjectName(_fromUtf8("tableWidget_1_13"))
        self.label_63 = QtGui.QLabel(self.tab_9)
        self.label_63.setGeometry(QtCore.QRect(540, 280, 211, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_63.setFont(font)
        self.label_63.setObjectName(_fromUtf8("label_63"))
        self.label_60 = QtGui.QLabel(self.tab_9)
        self.label_60.setGeometry(QtCore.QRect(10, 280, 181, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_60.setFont(font)
        self.label_60.setObjectName(_fromUtf8("label_60"))
        self.tableWidget_1_11 = QtGui.QTableWidget(self.tab_9)
        self.tableWidget_1_11.setGeometry(QtCore.QRect(10, 300, 400, 111))
        self.tableWidget_1_11.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.tableWidget_1_11.setObjectName(_fromUtf8("tableWidget_1_11"))
        self.tableWidget_1_9 = QtGui.QTableWidget(self.tab_9)
        self.tableWidget_1_9.setGeometry(QtCore.QRect(10, 20, 400, 111))
        self.tableWidget_1_9.setStyleSheet(_fromUtf8("background-color: rgb(170, 255, 255);"))
        self.tableWidget_1_9.setObjectName(_fromUtf8("tableWidget_1_9"))
        self.pushButton_3 = QtGui.QPushButton(self.tab_9)
        self.pushButton_3.setGeometry(QtCore.QRect(150, 470, 111, 41))
        self.pushButton_3.setStyleSheet(_fromUtf8("background-color: rgb(164, 157, 255);"))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton = QtGui.QPushButton(self.tab_9)
        self.pushButton.setGeometry(QtCore.QRect(150, 420, 111, 41))
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(164, 157, 255);"))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtGui.QPushButton(self.tab_9)
        self.pushButton_2.setGeometry(QtCore.QRect(280, 420, 111, 41))
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(164, 157, 255);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_4 = QtGui.QPushButton(self.tab_9)
        self.pushButton_4.setGeometry(QtCore.QRect(280, 470, 111, 41))
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(164, 157, 255);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_5 = QtGui.QPushButton(self.tab_9)
        self.pushButton_5.setGeometry(QtCore.QRect(410, 420, 111, 41))
        self.pushButton_5.setStyleSheet(_fromUtf8("background-color: rgb(164, 157, 255);"))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.label_64 = QtGui.QLabel(self.tab_9)
        self.label_64.setGeometry(QtCore.QRect(410, 480, 481, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_64.setFont(font)
        self.label_64.setObjectName(_fromUtf8("label_64"))
        self.tabWidget.addTab(self.tab_9, _fromUtf8(""))

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.RR_veriYukle1_btn.setText(_translate("Dialog", "1.Veri Yukle", None))
        self.RR_RveriOlustur1_btn.setText(_translate("Dialog", "Random Veri Yap", None))
        self.label.setText(_translate("Dialog", "Veri Miktari :", None))
        self.label_2.setText(_translate("Dialog", "Veri Aralik :", None))
        self.RR_RveriOlustur2_btn.setText(_translate("Dialog", "Random Veri Yap", None))
        self.label_3.setText(_translate("Dialog", "Veri Miktari :", None))
        self.RR_veriYukle2_btn.setText(_translate("Dialog", "2.Veri Yukle", None))
        self.label_4.setText(_translate("Dialog", "Veri Aralik :", None))
        self.RR_RUS_btn.setText(_translate("Dialog", "RUS", None))
        self.RR_ROS_btn.setText(_translate("Dialog", "ROS", None))
        self.label_5.setText(_translate("Dialog", "TUM VERILER", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "   RUS / ROS  ", None))
        self.KNN_veriYukle_btn.setText(_translate("Dialog", "Veri Yukle", None))
        self.KNN_YeniveriYukle_btn.setText(_translate("Dialog", "Yeni Veri Ekle", None))
        self.KNN_Kumeleme_btn.setText(_translate("Dialog", "Kumelemeyi Gerceklestir", None))
        self.label_8.setText(_translate("Dialog", "X", None))
        self.label_9.setText(_translate("Dialog", "Y", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "  K Nearest Neighborhood  ", None))
        self.Kmeans_veriYukle_btn.setText(_translate("Dialog", "Veri Yuke", None))
        self.Kmeans_Kumele_btn.setText(_translate("Dialog", "Kumele", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Dialog", "   K - MEANS   ", None))
        self.NavieBay_btn.setText(_translate("Dialog", "Navie Bayes Tahmni", None))
        self.label_12.setText(_translate("Dialog", "Kelime Giriniz", None))
        self.label_13.setText(_translate("Dialog", "Kategorisi", None))
        self.label_14.setText(_translate("Dialog", "?????????", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Dialog", "   NAVIE  BAYES   ", None))
        self.Norm_btn_MinMax.setText(_translate("Dialog", "Min-Max", None))
        self.Norm_btn_veriYukle.setText(_translate("Dialog", "Veri Yükle", None))
        self.Norm_btn_Zscore.setText(_translate("Dialog", "Z-Score", None))
        self.Norm_btn_Median.setText(_translate("Dialog", "Median", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("Dialog", "   NORMALIZASYON    ", None))
        self.label_6.setText(_translate("Dialog", "Tum veriler", None))
        self.label_7.setText(_translate("Dialog", "Train veriler", None))
        self.label_10.setText(_translate("Dialog", "Test veriler", None))
        self.TT_upload_btn.setText(_translate("Dialog", "Upload", None))
        self.label_11.setText(_translate("Dialog", "Train Percent :", None))
        self.TT_comboBox.setItemText(0, _translate("Dialog", "10", None))
        self.TT_comboBox.setItemText(1, _translate("Dialog", "20", None))
        self.TT_comboBox.setItemText(2, _translate("Dialog", "30", None))
        self.TT_comboBox.setItemText(3, _translate("Dialog", "40", None))
        self.TT_comboBox.setItemText(4, _translate("Dialog", "50", None))
        self.TT_comboBox.setItemText(5, _translate("Dialog", "60", None))
        self.TT_comboBox.setItemText(6, _translate("Dialog", "70", None))
        self.TT_comboBox.setItemText(7, _translate("Dialog", "80", None))
        self.TT_comboBox.setItemText(8, _translate("Dialog", "90", None))
        self.TT_uygula_btn.setText(_translate("Dialog", "Uygula", None))
        self.RanForest_uygula_btn.setText(_translate("Dialog", "Random Forest", None))
        self.label_16.setText(_translate("Dialog", "Basari orani :", None))
        self.label_17.setText(_translate("Dialog", "0", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_8), _translate("Dialog", "    TRAIN  -  TEST     ", None))
        self.label_61.setText(_translate("Dialog", "SST Test Veri Kümesi", None))
        self.label_59.setText(_translate("Dialog", "DST Train Veri Kümesi", None))
        self.label_58.setText(_translate("Dialog", "SST Train Veri Kümesi", None))
        self.label_62.setText(_translate("Dialog", "DST Test Veri Kümesi", None))
        self.pushButton_6.setText(_translate("Dialog", "Verileri \n"
" Yükle", None))
        self.label_63.setText(_translate("Dialog", "STCP Test Veri Kümesi", None))
        self.label_60.setText(_translate("Dialog", "STCP Train Veri Kümesi", None))
        self.pushButton_3.setText(_translate("Dialog", "DST", None))
        self.pushButton.setText(_translate("Dialog", "SST", None))
        self.pushButton_2.setText(_translate("Dialog", "STPC", None))
        self.pushButton_4.setText(_translate("Dialog", "Tümünü Ver", None))
        self.pushButton_5.setText(_translate("Dialog", "Random Forest", None))
        self.label_64.setText(_translate("Dialog", "Result", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_9), _translate("Dialog", "        PARKINSON       ", None))
