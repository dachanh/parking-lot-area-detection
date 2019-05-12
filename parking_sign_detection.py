import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2
import numpy as np


class parking_sign_detection(object):
    def __init__(self):
        self.detector = cv2.ORB_create()
        self.template =  cv2.imread('parking_sign.png',0)
        self.keypoint_1,self.description_1 = self.detector.detectAndCompute(self.template,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.MIN_SCORE = 20
    def isSquare(self, pts):
        sides = []
        sides.append(pts[1][0][1] - pts[0][0][1])
        sides.append(pts[2][0][0] - pts[1][0][0])
        sides.append(pts[2][0][1] - pts[3][0][1])
        sides.append(pts[3][0][0] - pts[0][0][0])
        maxSide = max(sides)
        minSide = min(sides)
        meanSide = np.mean(sides)
        if maxSide > meanSide*1.1 or minSide < meanSide *0.9 or meanSide < 1:
            return False
        return True
    def isParkingSign(self,img_src):
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
        keypoint_2,description_2 = self.detector.detectAndCompute(img,None) 
        print(description_2)
        matches = self.flann.knnMatch(np.asarray(self.description_1,np.float32),np.asarray(description_2,np.float32),k=2)
        good = [] 
        print(len(matches))
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        #print(len(good))
        if len(good) >= self.MIN_SCORE:
            source_points = np.float32([self.keypoint_1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            destination_points = np.float32([keypoint_2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M,mask = cv2.findHomography(source_points,destination_points,cv2.RANSAC,5.0)
            
        else:
            print("asdasds")
        

dect = parking_sign_detection()
img = cv2.imread('./parking_2.png')
dect.isParkingSign(img)