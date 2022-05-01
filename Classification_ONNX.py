import tensorflow as tf

import onnx

from onnx_tf.backend import prepare

model = onnx.load('model.onnx') ##load onnx model

tf_rep = prepare(model) ##

import numpy as np
from PIL import Image



###image acquisition and classification


import cv2 ##camera test capture on space

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    flag = True

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
        img_name = "capture/terrain.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        imgtest = Image.open('capture/terrain.png').resize((224, 224))
        imgtest = tf.transpose(imgtest)
        imgtest = tf.expand_dims(imgtest,0)
        with tf.Session() as sess:
            imgtest = sess.run(imgtest)
        acc = tf_rep.run(imgtest)
        val = np.argmax(acc)
        if val==2:print('grass')     #these values (0,1,2) refer to the classifier output. In this case the model returned a single digit number (0 1 2), change to your needs.
        elif val==0:print('asphalt')
        elif val==1:print('dirt')
    #lag=True    

cam.release()

cv2.destroyAllWindows()
