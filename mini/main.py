import cv2 as cv

background = cv.imread("background.png")
background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
background = cv.GaussianBlur(background, (21,21),0)

cap = cv.VideoCapture("test.avi")

#car_cascade = cv.CascadeClassifier('cars.xml')

while(cap.isOpened()):
	ret, frame = cap.read()
	
	#convert to grayscale
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	#apply blur to remove noise in frame
	gray = cv.GaussianBlur(gray, (21,21), 0)

	#check difference between background and current frame
	diff = cv.absdiff(background,gray)
	
	thresh = cv.thresh(diff, 30, 255, cv.THRESH_BINARY)[1]	
	thresh = cv.dilate(thresh, None, iterations=2)
	_, contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	
	for contour in contours:
		if cv.contourArea(contour)<10000:
			continue
		(x,y,w,h) = cv.boundingRect(contour)
		cv.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),3)

	#cars = car_cascade.detectMultiScale(gray, 1.1, 1)
	#for (x,y,w,h) in cars:
	#	cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 3)

	cv.imshow("Result", frame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break	

cap.release()
cv.destroyAllWindows()
