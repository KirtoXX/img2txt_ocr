from __future__ import print_function
import numpy as np
from keras.models import Model,Input
from scipy import misc
from img2txt import inference

def read_image(path):
    image = misc.imread(path)
    image = misc.imresize(image,[90,30])
    image = np.expand_dims(image,axis=0)
    return image

def decoder(logit):
    logit = np.reshape(logit,[7,16])
    str1 = ''
    loc = np.argmax(logit,axis=1)
    print(loc)
    for i in range(7):
        temp = loc[i]
        # decoder id
        for j in range(10):
            if temp==j:
                str1+=str(j)
        if temp==10:
            str1+='('
        elif temp==11:
            str1+=')'
        elif temp==12:
            str1+='+'
        elif temp==13:
            str1+='-'
        elif temp==14:
            str1+='*'

    return str1

def main():

    input = Input(batch_shape=[None,90,30,3])
    logit = inference(input)
    model = Model(input,logit)
    model.load_weights('weight/20171102-1.h5')

    path = 'test_image/1.png'
    image = read_image(path)
    result = model.predict(image)

    result = decoder(result)
    print(result)

if __name__ == '__main__':
    main()

