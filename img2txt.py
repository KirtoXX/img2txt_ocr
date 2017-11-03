from keras.layers import Conv2D,Activation,GRU
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dense



def inference(image_tensor):
    # the input size is 180*60*3
    x = Conv2D(32,(3,3),name='conv1')(image_tensor)
    x = MaxPool2D((2,2),name='pool1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)

    x = Conv2D(64,(3,3),name='conv2')(x)
    x = MaxPool2D((2, 2),name='pool2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu')(x)

    x = Conv2D(64,(3,3), name='conv3')(x)
    x = MaxPool2D((2, 2), name='pool3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu')(x)

    x = Flatten(name='flatte1')(x)

    x = Dense(256,name='fc1')(x)

    # max_length of sentence
    x = RepeatVector(7,name='repeat')(x)

    #2*GRU the output shape is (None,7,256)
    x = Bidirectional(GRU(128,name='GRU',return_sequences=True))(x)

    #apply (None,7,256) to all GRUs output (None,7,16)
    x = TimeDistributed(Dense(16,name='fc2',activation='softmax'))(x)
    x = Flatten(name='Flatten2')(x)

    return x





