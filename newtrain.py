# import model
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout, Activation, Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
# from visual_callbacks import AccLossPlotter
from keras.applications import MobileNet
from keras.models import Model
import numpy as np

img_baris,img_kolom = 224,224
MobileNet = MobileNet(weights='imagenet',
                     include_top=False,
                     input_shape=(img_baris,img_kolom,3))

for layer in MobileNet.layers:
    layer.trainable = False

for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+layer.__class__.__name__,layer.trainable)
    


def tambahModel(bottomModel,numClasses):
    topModel = bottomModel.output
    topModel = GlobalAveragePooling2D()(topModel)
    topModel = Dense(128,activation='relu')(topModel)
    topModel = Dense(128,activation='relu')(topModel)
    topModel = Dense(64,activation='relu')(topModel)
    topModel = Dense(numClasses,activation='softmax')(topModel)
    return topModel

numClasses = 2
FC_Head = tambahModel(MobileNet,numClasses)

model = Model(inputs = MobileNet.input,outputs = FC_Head)
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics= ['accuracy'])

TRAINING_DIR = "train"
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range=45,
                                  width_shift_range = 0.3,
                                  height_shift_range = 0.3,
                                  horizontal_flip=True,
                                  vertical_flip = True,
                                  zoom_range=0.25,
                                  shear_range=0.25,
                                  fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size=10, 
                                                    target_size=(224, 224))
VALIDATION_DIR = "test"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=10, 
                                                         target_size=(224, 224))
                                                         
checkpoint = ModelCheckpoint('Model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


history = model.fit_generator(train_generator,
                              epochs=30,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])

model.save_weights('makser1.h5')
model.save("maskerfix1.h5")