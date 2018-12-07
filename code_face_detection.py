import numpy as np
import cv2


#capturing video through webcam
cap=cv2.VideoCapture(0)

while True:

	#screen shot
	ret,frame=cap.read()
	
	
	#smoothening of image
	frame=cv2.GaussianBlur(frame,(11,11),0)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	
	
	#HSV thresholding of skin
	min=np.array([0,30,60])
	max_arr=np.array([20,150,255])
	mask=cv2.inRange(hsv,min,max_arr)
	
	
	#Morphological operations
	kernel=np.ones((9,9),np.uint8)
	mask=cv2.erode(mask,kernel,iterations=2)
	mask=cv2.dilate(mask,kernel,iterations=2)
	
	mask=cv2.dilate(mask,kernel,iterations=2)
	mask=cv2.erode(mask,kernel,iterations=2)
	
	
	#finding contours for similar colors 
	cnts=cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	
	#checking if face present
	if(len(cnts)>0):
		c=max(cnts,key=cv2.contourArea)
		((x,y),rad)=cv2.minEnclosingCircle(c)
		
		if rad>5:
			cv2.circle(frame,(int(x),int(y)),int(rad),(0,0,255),2)#drawing circle around the face
		
	
	cv2.imshow('original',frame)
	cv2.imshow('morphed',mask)
	
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
		
cv2.destroyAllWindows()
cap.release()
		