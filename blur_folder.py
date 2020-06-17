import cv2 as cv
import os

folder = input('Select a folder : ')
h_cascade_s = input('Select a Harr Cascade: ')
h_cascade = cv.CascadeClassifier(h_cascade_s)

with os.scandir(folder) as pics:
    for pic in pics:
        img = cv.imread(folder+'/'+pic.name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        roi = h_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in roi:
            f_blur = cv.medianBlur(img[y:y+h, x:x+w],33)
            img[y:y+h, x:x+w] = f_blur
        cv.imwrite('blurred_'+pic.name, img)
        print(pic.name + ' blurred')


