#! /usr/bin/env python
import numpy as np
import cv2

class TemplateMatching(object):
    def __init__(self):
        self.template = cv2.imread('./parking_sign.png',0)
        self.edged_template = self.auto_canny(template) 
    def auto_canny(self,img,sigma = 0.33):
        v = np.mean(img)
        lower = int(max(0,(1.0-sigma)*v))
        upper =int(min(255,(1.0 + sigma)*v))
        edged = cv2.Canny(img,lower,upper)
        return edged

    def detectAndCompute(self,img):
        query_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edged_query_img = self.auto_canny(query_img)
        h,w = edged_query_img.shape[0],edged_query_img.shape[1]
        #multi scale 
        for scale in np.linspace(0.2,1.0,20)[::-1]:
            resized = cv2.resize(gray,(w*scale,h),interpolation=cv2.INTER_CUBIC)
            result = cv2.matchTemplate(resized,self.edged_template,cv2.TM_CCOEFF)
            (_,maxVal,_maxLoc) = cv2.minMaxLoc(result)
            
