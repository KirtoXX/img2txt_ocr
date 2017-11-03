from __future__ import print_function
from keras.models import Model
from keras.models import Input
from img2txt import inference
import tensorflow as tf
from keras import backend as K
from keras.optimizers import sgd
import numpy as np
import shutil

def main():

    input = Input(batch_shape=(None,90,30,3))
    logit = inference(input)

    model = Model(input,logit)
    model.load_weights('weight/20171102.h5')
    model.summary()

    #tensorboard --logdir =logs
    #sess = K.get_session()
    #writer = tf.summary.FileWriter('logs/',sess.graph)

    X = np.load('npy_data/image.npy')
    y = np.load('npy_data/lable.npy')
    y = y.reshape([100000,7*16])
    print(X.shape)
    print(y.shape)

    opt = sgd(lr=0.1,decay=0.01,momentum=0.7)
    model.compile(optimizer='RMSprop',
                  #optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy','categorical_accuracy'])

    model.fit(x=X,y=y,batch_size=128,verbose=1,validation_split=0.4,epochs=300)
    model.save_weights('weight/20171102-1.h5')

if __name__ == '__main__':
    main()