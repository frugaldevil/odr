import cv2
import numpy as np
import nn_model
from preprocess import pad
import time as t

threshold = 0.99
model_name = "lemodel0.tfl"
left_border = 0.1
right_border = 0.9
font = cv2.FONT_HERSHEY_SIMPLEX

model = nn_model.generate_model()
model.load(model_name)
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    img = frame

    #im = cv2.Canny(img,100,200)
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    l = len(contours)
    m = 0

    for i in range(0, l):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(contours[i])
        dim_y, dim_x, depth = img.shape

        if(x >= (dim_x * left_border) and (x + w) <= (dim_x * right_border)):
            if((w < 300 and h < 300) and (w > 16 and h > 50)):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                sub_img = frame[y:(y + h), x:(x + w)]
                sub_img = pad(sub_img)
                sub_img = cv2.resize(sub_img, (32, 32))
                sub_img = np.array(sub_img, dtype="float16") / 256
                sub_img = sub_img.reshape([1, 32, 32, 3])
                pred = np.array(model.predict(sub_img),
                                dtype="float64")[0] ** 2
                index = np.argmax(pred)

                if(pred[index] > threshold):
                    digit = (index + 1) % 10
                    digit = str(digit)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                    img = cv2.putText(img, digit, (x, y + h + 25),
                                      font, 1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('image', img)

    k = cv2.waitKey(30)

    if(k == ord('x') or k == ord('X')):
        break
    elif(k == ord(' ')):
        cv2.imwrite(t.asctime(t.localtime()) + '.jpg', frame)

cap.release()
cv2.destroyAllWindows()
