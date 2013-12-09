import cv
import sys
import argparse 
import math

if len(sys.argv) != 3:
  sys.exit("Invalid arguments, using:python OpticalFlow.py path/to/image1 path/to/image2")

firstImage = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
secondImage = cv.LoadImage(sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE)

desImageHS = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
desImageLK = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)

#Using HS(Horn Schunk)
cols = firstImage.width
rows = firstImage.height

velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
vely = cv.CreateMat(rows, cols, cv.CV_32FC1)

cv.SetZero (velx)
cv.SetZero (vely)

STEP = 8

cv.CalcOpticalFlowHS(firstImage, secondImage, False, velx, vely, 100.0,(cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,64, 0.01))
for i in range(0, cols, STEP):
  for j in range(0, rows, STEP):
    dx = int(cv.GetReal2D (velx, j, i))
    dy = int(cv.GetReal2D (vely, j, i))
    cv.Line(desImageHS,(i, j),(i + dx, j + dy), (255, 0, 0), 1, cv.CV_AA, 0)

cv.SaveImage("resultHS.png", desImageHS)

#Using LK(Lukas Kanade)
desImageLK = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
desImageLK_100 = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
eignImg = cv.CreateImage(cv.GetSize(firstImage), cv.IPL_DEPTH_32F, 1)
derivateImg = cv.CreateImage(cv.GetSize(firstImage), cv.IPL_DEPTH_32F, 1)
features = cv.GoodFeaturesToTrack(firstImage, eignImg, derivateImg, 5000,  0.1, 10, None, True)

#Window size 50
r = cv.CalcOpticalFlowPyrLK(firstImage, secondImage, None, None, features, (50,50), 0, (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01) ,0)
list = r[0]
for i in range(len(list)) :
	dis = math.sqrt(math.pow((features[i][0]-list[i][0]),2) + math.pow((features[i][1]-list[i][1]),2))
	cv.Line(desImageLK, (int(features[i][0]), int(features[i][1])), (int(list[i][0]), int(list[i][1])), cv.CV_RGB(0, 0, 255), 1, cv.CV_AA, 0)
 
cv.SaveImage("resultLK_50.png", desImageLK)

#Window size 100
r = cv.CalcOpticalFlowPyrLK(firstImage, secondImage, None, None, features, (100,100), 0, (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01) ,0)
list = r[0]
for i in range(len(list)) :
	dis = math.sqrt(math.pow((features[i][0]-list[i][0]),2) + math.pow((features[i][1]-list[i][1]),2))
	cv.Line(desImageLK_100, (int(features[i][0]), int(features[i][1])), (int(list[i][0]), int(list[i][1])), cv.CV_RGB(0, 0, 255), 1, cv.CV_AA, 0)
 
cv.SaveImage("resultLK_100.png", desImageLK_100)

#Show result image
cv.NamedWindow("HS Result")
cv.ShowImage("HS Result", desImageHS)

cv.NamedWindow("LK Result, Window size = 50")
cv.ShowImage("LK Result, Window size = 50", desImageLK)

cv.NamedWindow("LK Result, Window size = 100")
cv.ShowImage("LK Result, Window size = 100", desImageLK_100)

#Quit window when ESC key is pressed
cv.WaitKey(0)
cv.DestroyAllWindows()
