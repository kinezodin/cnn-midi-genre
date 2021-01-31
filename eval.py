import data_utils as du
import numpy as np

from tensorflow.keras.callbacks import TensorBoard,Callback
import pandas as pd
import pypianoroll as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,multilabel_confusion_matrix
import os
from time import time
import matplotlib.pyplot as plt    
import matplotlib.patches as mpatches  
from sklearn.metrics import confusion_matrix,log_loss, roc_auc_score
import sys
import itertools
import models as mods
import train_utils as tu
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import math
from sklearn.metrics import roc_auc_score

_DATA_FOLDER='datasets/x_arrays/'
_TEST_DF_NAME = "TEST"
_REPORTS_FOLDER = 'reports/' 
_WEIGHTS_FOLDER = 'weights/'



def predict_song(seq_len,x,m):
	hop = seq_len//2
	n = x.shape[0]//hop-1

	x_slice = np.zeros((n,seq_len,128))

	for i in range(n):
		x_slice[i]=x[i*hop:i*hop+seq_len]
		x_slice[i] = x_slice[i]/(np.max(x_slice[i])+10e-7)
	p = m.predict(x_slice)
	p = np.average(p,axis=0)
	return p


def evaluate_model(df,sg,seq_len,m,mname):
	y_true=list()
	y_pred=list()

	for i in range(df.shape[0]):
		print(str(i/df.shape[0]))
		try:
			pi = np.load(_DATA_FOLDER+df['file_id'][i]+'.npy')
		except KeyboardInterrupt:
			sys.exit()
		except:
			continue
		y = np.zeros((len(sg,)))
		for j in range(len(sg)):
			y[j]=df_test[sg[j]][i]
		print(y)
		if not 2*pi.shape[0]//seq_len-1>0:
			continue
		yp = predict_song(seq_len,pi,m)
		if np.isnan(yp).any():
			continue
		print(yp)

		y_true.append(y)
		y_pred.append(yp)

	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	print(y_pred.shape)
	print(y_true.shape)

	report = roc_auc_score(y_true,y_pred)
	print(report)
	with open(_REPORTS_FOLDER+'AUC'+mname+'.txt','w') as f:
		f.write(str(report))

	report = classification_report(y_true,y_pred,target_names=sg, output_dict=True)
	df_report = pd.DataFrame(report).transpose()
	df_report.to_csv(_REPORTS_FOLDER+'cl-report-'+mname+'.csv')



dsets = ['masd','topmagd']
super_genre_dic=dict()
super_genre_dic['magd']=['Pop_Rock','Electronic','Country','RnB','Jazz','Latin','International','Rap','Vocal','New Age','Folk','Reggae','Blues']
super_genre_dic['topmagd']=['Pop_Rock','Electronic','Country','RnB','Jazz','Latin','International','Rap','Vocal','New Age','Folk','Reggae','Blues']
super_genre_dic['masd']= ['Big_Band', 'Blues_Contemporary','Country_Traditional','Dance','Electronica','Experimental','Folk_International','Gospel','Grunge_Emo','Hip_Hop_Rap','Jazz_Classic','Metal_Alternative','Metal_Death','Metal_Heavy','Pop_Contemporary','Pop_Indie','Pop_Latin','Punk','Reggae','RnB_Soul','Rock_Alternative','Rock_College','Rock_Contemporary','Rock_Hard','Rock_Neo_Psychedelia']


for d in dsets:
	super_genres=super_genre_dic[d]
	df_test=pd.read_csv(d+_TEST_DF_NAME+'.csv')
	num_classes = len(super_genres)
	bs=26
	lr=10e-5
	for seq_len in [64,128,256,512,1024,2048]:
		for tree in [True,False]:
			for shallow in [False,True]:
				if is_tree:
					prefix='TREE'
				else:
					prefix='SEQ'

				prefix=d+'_'+prefix
				m = mods.gen_model(seq_len,num_classes,lr=10e-5,chroma=False,drumz = False,shallow=shallow,tree=tree)
				
				mname = 'tree'+str(tree)+'shallow'+str(shallow)+prefix+str(seq_len)+''+str(lr)+'LR'
				m.load_weights(_WEIGHTS_FOLDER+mname+'.h5')
				evaluate_model(df_test,super_genres,seq_len,m,mname)
