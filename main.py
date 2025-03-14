import cv2
import cvzone

#Setup capture and settings Window for live webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height

def noop(a):
    pass

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings",640,240)
cv2.createTrackbar("Lower Value for Threshold","Settings",54,255,noop) # bar from 0-255 starting at 140
cv2.createTrackbar("Min Area for Contours","Settings",200,1000,noop) # bar from 0-255 starting at 140


def filter_contours(contours, min_area=5, max_area = 300000):
    filtered_contours = []  # Contours above min_area
    removed_contours = []  # Contours below min_area

    for contour in contours:
        if min_area <= cv2.contourArea(contour) <= max_area:
            filtered_contours.append(contour)
        else:
            removed_contours.append(contour)

    return tuple(filtered_contours), tuple(removed_contours)

def write_contour_to_file(label,contours,file):
    for contour in contours:
        file.write(label + str(cv2.contourArea(contour)) + '\n')


def draw_contour_area(img, contours):
    for contour in contours:
        area = cv2.contourArea(contour)
        # Calculate centroid for text placement
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = contour[0][0][0], contour[0][0][1]  # Use first contour point as fallback

        # Put area text on the image
        cv2.putText(img, f"{int(area)}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) # opacity=0.5, color=blue, width=1


def find_contours(imgThresh, img):
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cv2.getTrackbarPos("Min Area for Contours", "Settings")
    filtered_contours, removed_contours = filter_contours(contours, min_area)

    with open("contours.txt", "w") as f:
        write_contour_to_file("filtered: ", filtered_contours, f)
        write_contour_to_file("removed:  ", removed_contours, f)

    contours_img = img.copy()
    cv2.drawContours(contours_img, filtered_contours,-1,(0,255,0),1)
    cv2.drawContours(contours_img, removed_contours,-1,(0,0,255),1)

    draw_contour_area(contours_img, filtered_contours)

    return contours_img

def pre_processing(img):
    #blur
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    return imgPre

def threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerV = cv2.getTrackbarPos("Lower Value for Threshold", "Settings")
    lower=(0,0,lowerV)
    upper=(255,255,255)
    thresh = cv2.inRange(hsv,lower,upper)
    return thresh

while True:
    #success, img = cap.read() #live webcam (disabled)
    img = cv2.imread('/home/connorc/PycharmProjects/Thames/images/bees1.jpg') #using static image instead

    imgPre = pre_processing(img)
    imgThresh = threshold(imgPre)
    imgContours = find_contours(imgThresh, img)

    imgStacked = cvzone.stackImages([img,imgPre,imgThresh,imgContours],3,2)    # columns, row
    cv2.imshow("Image", imgStacked)

    #break when escape is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break