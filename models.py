from tensorflow.keras import backend as K
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import UpSampling1D,Conv1D,Input,Flatten, LSTM, Dense,TimeDistributed,Lambda,Multiply,Add,Concatenate,Bidirectional,Conv2D,MaxPooling2D, Reshape,Average, MaxPooling1D,AveragePooling2D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.optimizers import Adam
import train_utils as tu
import numpy as np
from sklearn.metrics import confusion_matrix,log_loss, roc_auc_score,classification_report


def gen_model(seq_len,num_classes,lr=10e-5,chroma=False,drumz = False,shallow=False,tree=False):

	def pooltree(inp):
		levels= list()
		levels.append(inp)
		for i in range(int(np.log2(seq_len))-1):
			levels.append(AveragePooling1D(2)(levels[-1]))
		return levels

	def shallow_block(inp):
		c = Conv1D(128,24,activation='relu',padding='same')(inp)
		c = MaxPooling1D(2)(c)
		return c

	def deep_block(inp):
		c = Conv1D(117,9,activation='relu',padding='same')(inp)
		c = Conv1D(117,9,activation='relu',padding='same')(c)
		c = Conv1D(128,9,activation='relu',padding='same')(c)
		c = MaxPooling1D(2)(c)
		return c


	inpu = Input((seq_len,128))
	
	curr_inp = inpu

	if tree:
		inps = pooltree(inpu)

	for i in range(int(np.log2(seq_len))-1):
		if shallow:
			o = shallow_block(curr_inp)
		else:
			o = deep_block(curr_inp)
		if tree:
			o = Concatenate(axis=-1)([o,inps[i+1]])
		curr_inp = o
		
	o = Flatten()(o)
	d = Dense(128,activation='relu')(o)
	d = Dense(num_classes,activation='sigmoid')(d)
	m = Model(inpu,d)
	opt = Adam(lr=lr)
	m.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc',tu.f1_m,tu.precision_m, tu.recall_m])
	m.summary()
	return m
