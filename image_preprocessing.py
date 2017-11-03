from scipy import misc
import numpy as np
from keras.utils import to_categorical


def preprocessing(image_path):
    data = misc.imread(image_path)
    data = np.expand_dims(data,axis=0)
    return data

def str_to_id(str1):
    m = 0
    for i in range(10):
        if str1 == str(i):
            m = i
    if str1 == "(":
        m = 10
    elif str1 == ")":
        m = 11
    elif str1 == "+":
        m = 12
    elif str1 == "-":
        m = 13
    elif str1 == "*":
        m = 14
    return m

def line_to_ids(line):
    #<eof> is 15 total number is max_length is 7
    result = np.ones(7)
    result = result*15
    for i in range(len(line)):
        result[i] = str_to_id(line[i])
    result = to_categorical(result,num_classes=16)
    return result


def main():
    lable_dir = 'data/labels.txt'
    image_dir = 'f:/baidu_ocr_data/image/'
    f = open(lable_dir)
    content = []
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            string, y = line.split()
            temp = list(string)
            content.append(temp)
        else:
            break
    f.close()

    #test
    print(content[1])
    temp = line_to_ids(content[1])
    print(temp)

    # convert lable to npy
    lable_npy = np.zeros([len(content),7,16])
    print(len(content))
    for i in range(len(content)):
        one_hot = line_to_ids(content[i])
        one_hot = np.expand_dims(one_hot,axis=0)
        lable_npy[i] = one_hot

    print('lable finish')
    print(lable_npy.shape)
    np.save('npy_data/lable.npy',lable_npy)

    # convert image to npy
    image_npy = np.zeros([len(content),90,30,3])
    for i in range(len(content)):
        print(i)
        image_path = image_dir+str(i)+'.png'
        image = misc.imread(image_path)
        image = misc.imresize(image,size=[90,30])
        image = np.expand_dims(image,axis=0)
        image_npy[i] = image

    print('image finish!')
    print(image_npy.shape)
    np.save('npy_data/image.npy',image_npy)



if __name__ == '__main__':
    main()



