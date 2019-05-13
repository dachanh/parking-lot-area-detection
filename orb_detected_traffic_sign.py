import cv2
import numpy as np


class parking_sign_detector(object):
    def __init__(self):
        self.template = cv2.imread('./parking_sign.png',0)
        self.dectector =cv2.ORB_create()
        self.kps_1 , self.des_1 = self.dectector.detectAndCompute(self.template,None)
        self.bf = cv2.BFMatcher()
        self.GOOD_SCORES = 15
    def find_contour(self,img):
        temp = img.copy()
        _,contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True)
            area = cv2.contourArea(cnt)
            rect = cv2.boundingRect(cnt)
            if (len(approx) > 8) & (len(approx) < 15) & (area > 30) & (rect[2] < 300) & (rect[3] < 300) & (rect[2] >= 30) or (rect[3] >= 30):
                x,y,w,h = rect
                candidates.append([x,y,w,h])
        return candidates
    def blue_extraction(self,img):
        img = cv2.GaussianBlur(img,(5,5),0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue =  cv2.inRange(hsv, np.array([100,150,0]), np.array([140,255,255]))
        kernel = np.ones((3,3),np.uint8)
        close = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
        openc = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
        openc = cv2.GaussianBlur(openc,(9,9),2,2)
        return openc 
    def matching_points(self,img):
        temp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,temp =cv2.threshold(temp,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
        kps_2 , des_2 = self.dectector.detectAndCompute(temp,None)
        matches = self.bf.knnMatch(des_2,self.des_1,k=2)
        counting_scores = 0 
        for m , n in matches:
            if m.distance < 0.75*n.distance : counting_scores +=1
        return counting_scores

    def detectAndCompute(self,img):
        temp =img.copy()
        temp = self.blue_extraction(temp)
        cnts = self.find_contour(temp)
        for cnt in cnts:
            x,y,w,h = cnt
            temp = img[y:y+h,x:x+w] 
            cv2.imshow("sdsd",temp)
            cv2.waitKey(0)
            scores = self.matching_points(temp) 
            print("score",scores)
            if scores >= self.GOOD_SCORES:
                print("Parking Sign")
            else:
                print("No Parking Sign")


#detector = parking_sign_detector()
#detector.detectAndCompute(cv2.imread('./sign.png'))
