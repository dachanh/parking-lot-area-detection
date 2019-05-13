#! /usr/bin/env python
import numpy as np
import glob
import os
import cv2

class Template_matching(object):
    def __init__(self):
        self.template = cv2.imread('./parking_sign.png',0) 
    def resize(self,image, width = None, height = None, inter = cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized
    def detectAndCompute(self,img):
        (tH, tW) = self.template.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(3,3),0)
        gray = cv2.Laplacian(blur,cv2.CV_64F)
        gray = np.float32(gray)
        found = None
        for scale in np.linspace(0.5, 2, 30):
            resized = self.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            result = cv2.matchTemplate(resized, self.template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        if found is not None:
            #print(imagePath,'maxval=',maxVal,'Cursor detected at location:',(int(maxLoc[0] * r), int(maxLoc[1] * r)))
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.imshow('img',img)
            return (1,[startX,startY,endX,endY])
        return (-1,None)


detector = Template_matching()
img = cv2.imread('./parking_2.png')
res,loc= detector.detectAndCompute(img)
print(res)

