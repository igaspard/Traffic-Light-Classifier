from keras.models import load_model
import numpy as np
import cv2
import sys
import time
import os
import urllib

def main():
    if len(sys.argv) < 2:
        print("Usage need picture path")
        sys.exti(1)

    PWD = os.path.abspath(os.path.dirname(__file__))

    model_path = PWD + '/model.h5'

    model_link = 'https://s3-us-west-1.amazonaws.com/carndmodel/model.h5'

    if not os.path.exists(model_path):
        print('Need download the models from AWS')
        urllib.urlretrieve(model_link, model_path)
    else:
        print('models from AWS existed, good to go')

    path = sys.argv[1]
    start = time.time()
    model = load_model('model.h5')
    end = time.time()
    print('Load model time', end - start)
    model.summary()

    #test1_img = cv2.imread('trafficlight_images/1_image306_resize.jpg')
    test1_img = cv2.imread(path)
    test1_img = cv2.resize(test1_img, (100, 200), interpolation=cv2.INTER_CUBIC)
    test1_img = np.asarray(test1_img) / 255
    test = np.array([test1_img,])

    start = time.time()
    test_pred = model.predict(test)
    end = time.time()
    print('Predict time', end - start)
    test_pred = np.round(test_pred)

    print(test_pred)

    if test_pred[0][0] == 1.0:
        print(path, 'is Red Light')
    elif test_pred[0][1] == 1.0:
        print(path, 'is Yellow Light')
    elif test_pred[0][2] == 1.0:
        print(path, 'is Green Light')
    else:
        print(path, 'is unknown???')


if __name__ == '__main__':
    main()
