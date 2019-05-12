import cv2
import imutils
import numpy as np

def get_feature(candidate,template):
    dectector =cv2.ORB()
    keyPoint_template , descriptior_template = dectector.detectAndCompute(template,None)
    keyPoint_candidate , descriptior_candidate = dectector.detectAndCompute(candidate,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptior_candidate,descriptior_template,k=2)
    countGoodPoints = 0 
    for m , n in matches:
        if (m.distance < 0.75*n.distance):
            countGoodPoints +=1
    return countGoodPoints

def detected_traffic_sign(img,img_LeftSign,img_RightSign):
    candidate_width , candidate_height = img.shape[:2]
    template_LeftSign_width , template_LeftSign_height = img_LeftSign.shape[:2]
    template_RightSign_width , template_RightSign_height = img_RightSign.shape[:2]
    average_width = int((candidate_width + template_LeftSign_width + template_RightSign_width)/3)
    average_height = int((candidate_height + template_LeftSign_height + template_RightSign_height)/3)
    img_LeftSign = cv2.resize(img_LeftSign,(average_height,average_width))
    img_RightSign = cv2.resize(img_RightSign,(average_height,average_width))
    img = cv2.resize(img,(average_height,average_width))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = 0
    goodPointsLeftSign = get_feature(img,img_LeftSign)
    goodPointsRightSign = get_feature(img,img_RightSign)
    if (goodPointsLeftSign < goodPointsRightSign):
        result = 1 
    else:
        if (goodPointsLeftSign > goodPointsRightSign):
            result = 2
        else:
            result = 0
    return result


def find_contour(img):
    temp = img.copy()
    _,contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True)
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        if (len(approx) > 8) & (len(approx) < 15) & (area > 30) & (rect[2] < 300) & (rect[3] < 300) & (rect[2] >= 30) or (rect[3] >= 30):
            x,y,w,h = rect
            candidates.append([x,y,w,h])
    return candidates
def fill_hole(img):
    img_floodfill = img.copy()
    h , w  = img_floodfill.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(img_floodfill,mask,(0,0),255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    img_out = img | img_floodfill_inv
    return img_out
def removeSmallComponents(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape),dtype = np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2
def blue_extraction(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue =  cv2.inRange(hsv, np.array([100,150,0]), np.array([140,255,255]))
    kernel = np.ones((3,3),np.uint8)
    close = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    openc = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    openc = cv2.GaussianBlur(openc,(9,9),2,2)
    return openc

def main():
    cap =cv2.VideoCapture("./localization_traffic_sign/1.mp4")
    img_LeftSign = cv2.imread('/parking_sign.png')
    numframe = -1 
    while(cap.isOpened()):
        _,img = cap.read()
        numframe = numframe + 1  
        frame_1 = blue_extraction(img)
        frame_1 = removeSmallComponents(frame_1,50)
        frame_1 = fill_hole(frame_1)
        #cv2.imshow("frame 1",frame_1)
        candidates = find_contour(frame_1)
        #print(candidates)
        count = 0
        for candidate in candidates:
            print(candidate)
            count = count + 1
            x,y,w,h = candidate[:4]
            subImg = img[y:y+h,x:x+w,]
            nameOfTrafficSign = detected_traffic_sign(subImg,img_LeftSign,img_RightSign)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255, 255, 00), 2)
            if nameOfTrafficSign == 1:
                text ="Right Sign"
            else:
                if nameOfTrafficSign == 0 :
                    text = "Left Sign"
                else:
                    text = "unidentified"
            cv2.putText(img, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
        img = cv2.resize(img,(800,800))
        cv2.imshow("result",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()