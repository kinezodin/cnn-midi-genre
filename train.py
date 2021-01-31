import data_utils as du
import train_utils as tu
import models as mods


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
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model



_DATA_FOLDER='datasets/x_arrays/'
_TRAIN_DF_NAME = "TRAIN"
_VAL_DF_NAME = "TEST" 
_LOGDIR = './logs/'
_WEIGHTS_FOLDER = 'weights/'

## Data generator to use while training
def npy_dg(df,seq_len,bs=8,is_test=False,balance=False,is_tree=False,super_genres=['Pop_Rock','Electronic','Country','RnB','Jazz','Latin','International','Rap','Vocal','New Age','Folk','Reggae','Blues']):

	while True:
		batch_all_df = df.sample(n=bs//2).reset_index(drop=True)
		batch_noprock=noprock.sample(n=bs//2).reset_index(drop=True)
	
		xs=list()
		ys=list()
		#for batch_df in batch_all_df:
		x = np.zeros((bs//2,seq_len,128))
		if drumz:
			x_dr = np.zeros((bs/2,seq_len,128))
		y = np.zeros((bs//2,len(super_genres)))
		for i in range(batch_df.shape[0]):

			try:
				pi = np.load(_DATA_FOLDER+batch_df['file_id'][i]+'.npy')
				if drumz:
					dr = np_file['dr']
			except KeyboardInterrupt:
				sys.exit()
			except:
				i-=1
				continue
			if pi.shape[0]<seq_len:
				x[i,:pi.shape[0]]=pi
				if drumz:
					x_dr[i,:pi.shape[0]]=dr
			else:
				strt = np.random.randint(pi.shape[0]-seq_len)
				x[i] = pi[strt:strt+seq_len]
				
				if not is_test:
					transp = np.random.randint(-9,9)
					tr = pp.Track(pianoroll=x[i])
					tr.transpose(transp)
					x[i] = tr.pianoroll
					if is_tree:
						x[i] = du.pianoroll_to_halftree(x[i])
				for j in range(len(super_genres)):
					y[i,j]=batch_df[super_genres[j]][i]
				x[i] = x[i]/(np.max(x[i],axis=1).reshape((seq_len,1))+10e-7)
				
		xs.append(x)
		ys.append(y)

		x = np.concatenate(xs,axis=0)
		y = np.concatenate(ys,axis=0)

		yield x,y

dsets = ['masd','topmagd']
super_genre_dic=dict()
super_genre_dic['magd']=['Pop_Rock','Electronic','Country','RnB','Jazz','Latin','International','Rap','Vocal','New Age','Folk','Reggae','Blues']
super_genre_dic['topmagd']=['Pop_Rock','Electronic','Country','RnB','Jazz','Latin','International','Rap','Vocal','New Age','Folk','Reggae','Blues']
super_genre_dic['masd']= ['Big_Band', 'Blues_Contemporary','Country_Traditional','Dance','Electronica','Experimental','Folk_International','Gospel','Grunge_Emo','Hip_Hop_Rap','Jazz_Classic','Metal_Alternative','Metal_Death','Metal_Heavy','Pop_Contemporary','Pop_Indie','Pop_Latin','Punk','Reggae','RnB_Soul','Rock_Alternative','Rock_College','Rock_Contemporary','Rock_Hard','Rock_Neo_Psychedelia']


for d in dsets:
	super_genres=super_genre_dic[d]
	df_train=pd.read_csv(d+_TRAIN_DF_NAME+'.csv')
	df_val=pd.read_csv(d+_VAL_DF_NAME+'.csv')
	num_classes = len(super_genres)
	
	bs=26
	lr=10e-5
	for seq_len in [64,128]:
		for tree in [True,False]:
			for shallow in [True,False]:
				if is_tree:
					prefix='TREE'
				else:
					prefix='SEQ'


				prefix=d+'_'+prefix
				m = mods.gen_model(seq_len,num_classes,lr=10e-5,chroma=False,drumz = False,shallow=shallow,tree=tree)

				dag_train = npy_dg_simpuru(df=df_train,seq_len=seq_len,is_tree=tree,bs=bs,super_genres=super_genres,balance=False)
				dag_val = npy_dg_simpuru(df=df_val,seq_len=seq_len,is_test=True,is_tree=tree,bs=bs,super_genres=super_genres)

				tb = TensorBoard(log_dir=_LOGDIR+'tree-'+str(tree)+'shallow-'+str(shallow)+prefix+str(seq_len)+''+str(lr)+'LR', histogram_freq=0, batch_size=bs, write_graph=True, write_grads=False)
						

				m.fit_generator(dag_train,validation_data=dag_val,steps_per_epoch=1000,epochs=300,callbacks=[tb,cm,rauc],validation_steps=100,initial_epoch=0)

				m.save_weights(_WEIGHTS_FOLDER+'tree'+str(tree)+'shallow'+str(shallow)+prefix+str(seq_len)+''+str(lr)+'LR.h5')
				
				del m
				K.clear_session()
