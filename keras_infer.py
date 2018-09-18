from keras.models import load_model
import numpy as np
import cv2
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage need picture path")
        sys.exti(1)

    path = sys.argv[1]

    model = load_model('model.h5')
    model.summary()

    #test1_img = cv2.imread('trafficlight_images/1_image306_resize.jpg')
    test1_img = cv2.imread(path)
    test1_img = np.asarray(test1_img) / 255
    test = np.array([test1_img,])

    test_pred = model.predict(test)
    test_pred = np.round(test_pred)

    print(test_pred)

if __name__ == '__main__':
    main()
