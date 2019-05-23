#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:32:35 2019
@author: kui
"""

from keras.layers import merge,Conv2D,ZeroPadding2D,Input,BatchNormalization,Activation,MaxPooling2D,UpSampling2D,Deconv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from PIL import Image
import numpy as np
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.merge import concatenate,add,Dot,multiply
from keras.layers.core import Dense, Dropout, Reshape
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

K.set_image_data_format('channels_first')
smooth = 1.0
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def conv_block(ip, nb_filter, dropout_rate, weight_decay):

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
       
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    
    return x


      
def Decon_stage(x,nb_filters,kernel_size=(3,3),strides=(2,2),weight_decay=5e-4):

    x = Deconv2D(nb_filters,kernel_size,strides=strides, activation='relu', padding='same',data_format='channels_first',
                        kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

#    x = UpSampling3D()(x)   
    return x

def output(x,weight_decay):

    x = Conv2D(1, (1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('sigmoid')(x)
    return x


def side_out(x,nb_filters,kernel_size=(3,3),strides=(2,2),weight_decay=5e-4):
    up = Decon_stage(x,nb_filters,kernel_size,strides,weight_decay)
    out = output(up,weight_decay)

    return out
   
img_rows = 512
img_cols = 512
 
def NetWork(dropout_rate=0.3, weight_decay=5e-4):

   
    if  K.image_data_format() == 'channels_last':
      img_input = Input(shape=(img_rows,img_cols,1))

    else:
      img_input = Input(shape=(1,img_rows, img_cols))



    # Initial convolution

    x0_0 = conv_block(img_input, 64, dropout_rate=0.3, weight_decay=5e-4)
    x0_1 = MaxPooling2D((2,2))(x0_0)
    print('x:',K.int_shape(x0_1))
    
    
    x1_0 = conv_block(x0_1, 128, dropout_rate=0.3, weight_decay=5e-4)
    x1_1 = MaxPooling2D((2,2))(x1_0)
    print('x:',K.int_shape(x1_1))    
    
    
    x2_0 = conv_block(x1_1, 256, dropout_rate=0.3, weight_decay=5e-4)
    x2_1 = MaxPooling2D((2,2))(x2_0)
    print('x:',K.int_shape(x2_1))    
    
    
    x3_0 = conv_block(x2_1, 512, dropout_rate=0.3, weight_decay=5e-4)
    x3_1 = MaxPooling2D((2,2))(x3_0)
    print('x:',K.int_shape(x3_1))


    x4_0 = conv_block(x3_1, 512, dropout_rate=0.3, weight_decay=5e-4)
    print('x:',K.int_shape(x4_0))    
    



    up0 = Decon_stage(x4_0,512,kernel_size=(3,3),strides=(2,2),weight_decay=5e-4)
    add0 =add([up0,x3_0])
    x5_0 = conv_block(add0, 256, dropout_rate=0.3, weight_decay=5e-4)    
    
    
    up1 = Decon_stage(x5_0,256,kernel_size=(3,3),strides=(2,2),weight_decay=5e-4)
    add1 =add([up1,x2_0])
    x6_0 = conv_block(add1, 128, dropout_rate=0.3, weight_decay=5e-4)     

    up2 = Decon_stage(x6_0,128,kernel_size=(3,3),strides=(2,2),weight_decay=5e-4)
    add2 =add([up2,x1_0])
    x7_0 = conv_block(add2, 64, dropout_rate=0.3, weight_decay=5e-4)     
   
    up3 = Decon_stage(x7_0,64,kernel_size=(3,3),strides=(2,2),weight_decay=5e-4)
    add2 =add([up3,x0_0])
    x8_0 = conv_block(add2, 32, dropout_rate=0.3, weight_decay=5e-4)         
    
    
    
    
  
    output0 = output(x0_0, weight_decay)
    
    output1 = side_out(x1_0,32,kernel_size=(3,3),strides=(2,2))
    
    output2 = side_out(x2_0,32,kernel_size=(3,3),strides=(4,4))
    
    output3 = side_out(x3_0,32,kernel_size=(3,3),strides=(8,8))
    
    output4 = side_out(x4_0,32,kernel_size=(3,3),strides=(16,16))
    
    output5 = side_out(x5_0,32,kernel_size=(3,3),strides=(8,8))    
    
    output6 = side_out(x6_0,32,kernel_size=(3,3),strides=(4,4))

    output7 = side_out(x7_0,32,kernel_size=(3,3),strides=(2,2))
       
    output8 = output(x8_0, weight_decay)


    print(K.int_shape(output0))

    print(K.int_shape(output1))

    print(K.int_shape(output2))

    print(K.int_shape(output3))

    print(K.int_shape(output4))

    print(K.int_shape(output5))

    print(K.int_shape(output6))

    print(K.int_shape(output7))    

    print(K.int_shape(output8))   

   
    model = Model(img_input, [output0, output1, output2, output3, output4, output5, output6, output7, output8])
  #  plot_model(model, to_file='Vnet.png',show_shapes=True)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=dice_coef_loss,metrics=[dice_coef])#'binary_crossentropy'loss_weights=[0.3,0.6,1.0]
    return model

def preprocess(imgs):
    imgs=imgs.reshape(imgs.shape[0],1,imgs.shape[-2],imgs.shape[-1])
    return imgs
   
def preprocess_ge(imgs):
    imgs=imgs.reshape(imgs.shape[0],imgs.shape[-2],imgs.shape[-1])
    return imgs
   
   

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    data_path='/TrainingAndTesting/'
   
    test_x = np.load(data_path+'test_x.npy',encoding='latin1')
    test_y = np.load(data_path+'test_y.npy',encoding='latin1')

    train_x = np.load(data_path+'train_x.npy',encoding='latin1')
    train_y = np.load(data_path+'train_y.npy',encoding='latin1')   
   

    print('trainsamples',train_x.shape)
    print('testsamples',test_x.shape)
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = NetWork()
    model_checkpoint = ModelCheckpoint('Deep_Supervised_network.hdf5', monitor='val_loss', save_best_only=True)
   # early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(train_x,
              [train_y]*9,
              batch_size=8,
              nb_epoch=1000,
              verbose=2,
              shuffle=True,
              validation_data=(test_x,[test_y]*9),callbacks=[model_checkpoint])

if __name__ == '__main__':
        train_and_predict()
#    model = discriminator_model()
 