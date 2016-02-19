from glob import glob

import cv2
import numpy as np
import time
import timeit
from threading import Thread
import imutils
from collections import deque

# kriterijum zaustavljanja
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objp1 = np.zeros((6*9, 3), np.float32)
objp1[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# liste u kojima ce se cuvati dobijeni podatci
objpoints = []  # 3d tacke u realnom prostoru
imgpoints = []  # 2d tacke na slici

objpoints1 = []  # 3d tacke u realnom prostoru
imgpoints1 = []  # 2d tacke na slici

#images = glob.glob('*.jpg') #koliko shvatam, ovde uzima sliku... Ovo ce biti izmenjeno
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# promenljive vazne za kalibraciju
ret = 0
mtx = 0
dist = 0
rvecs = 0
tvecs = 0
imgShape = 0

ret1 = 0
mtx1 = 0
dist1 = 0
rvecs1 = 0
tvecs1 = 0
imgShape1 = 0

newcameramtx = 0
roi = 0

newcameramtx1 = 0
roi1 = 0


def calculateCoefs():
	global ret
	global mtx
	global dist
	global rvecs
	global tvecs
	global imgShape
	start = timeit.default_timer()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgShape, None, None)
	stop = timeit.default_timer()
	print 'Sracunati koef kamere 0!'
	print 'Proracun za kameru 0 je trajao: ' + str(stop-start)


def calculateCoefs1():
	global ret1
	global mtx1
	global dist1
	global rvecs1
	global tvecs1
	global imgShape1
	start = timeit.default_timer()
	ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, imgShape1, None, None)
	stop = timeit.default_timer()
	print 'Sracunati koef kamere 1!'
	print 'Proracun za kameru 1 je trajao: ' + str(stop-start)

def undistortTheDamnThing(img):
	global mtx
	global dist
	global newcameramtx
	global roi

	#undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	#crop the image
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]

	return dst  # dst ti je slika koja nije distortovana

def undistortTheDamnThing1(img1):
	global mtx1
	global dist1
	global newcameramtx1
	global roi1

	#undistort
	dst = cv2.undistort(img1, mtx1, dist1, None, newcameramtx1)

	#crop the image
	x, y, w, h = roi1
	dst = dst[y:y+h, x:x+w]

	return dst  # dst ti je slika koja nije distortovana


def proracunGreskeKompenzacije():
	mean_error = 0
	global objpoints
	global objpoints
	global rvecs
	global tvecs
	global mtx
	global dist
	global imgpoints
	for i in xrange(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		tot_error += error

	print 'total error: ' + mean_error/len(objpointsp)


def refineImage(img):
	global newcameramtx
	global roi
	global mtx
	global dist
	global imgShape
	# h, w = img.shape[:2]    # zamenjeno jednostavnim imgShape jer je to uvek konstantno, samo ga trebas azurirati nakon kalibracije posto se tu menja
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imgShape, 1, imgShape)

def refineImage1(img):
	global newcameramtx1
	global roi1
	global mtx1
	global dist1
	global imgShape1
	# h, w = img.shape[:2]    # zamenjeno jednostavnim imgShape jer je to uvek onstantno
	newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, imgShape1, 1, imgShape1)

while True:
	ret0, frame0 = cap0.read()
	ret1, frame1 = cap1.read()

	frame0calib = np.copy(frame0)
	frame1calib = np.copy(frame1)

	cv2.circle(frame0calib, (320, 240), 10, (255, 0, 0), 3)
	cv2.circle(frame1calib, (320, 240), 10, (255, 0, 0), 3)

	cv2.imshow('prikaz kamere 0', frame0calib)
	cv2.imshow('prikaz kamere 1', frame1calib)

	keyPressed = cv2.waitKey(1) & 0xFF

	if keyPressed == ord('c'):   # pritiskom na taster 'c' trazi se sahovnica na trenutnom frejmu
		gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
		gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

		# nadji ivice sahovnice
		ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
		ret1, corners1 = cv2.findChessboardCorners(gray1, (9, 6), None)
		# ukoliko su ivice nadjene, dodaj tacke objekta i slike (nakon njihovog rafiniranja)
		if ret == True:
			#cv2.imshow('Uhvacena slika', gray)
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
			imgpoints.append(corners2)

			# nacrtaj i prikazi ivice
			img = cv2.drawChessboardCorners(frame0, (9, 6), corners2, ret)

			cv2.imshow('Slika 0 sa oznacenim ivicama', img)
		# radimo sa dva if-a da bismo mogli da kalibrisemo coskove kamera
		if ret1 == True:
			objpoints1.append(objp1)
			corners12 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
			imgpoints1.append(corners12)

			img1 = cv2.drawChessboardCorners(frame1, (9, 6), corners12, ret1)

			cv2.imshow('Slika 1 sa oznacenim ivicama', img1)

		if ret is True | ret1 is True:
			cv2.waitKey(1)

	if keyPressed == ord('p'):   # kada si sve snimio sto te zanima, stisni 'p' da bi ti sracunao matricu kamere i koeficijente distorzije
		imgShape = gray.shape[::-1]
		imgShape1 = imgShape
		break

calculateCoefs()
calculateCoefs1()

time.sleep(2)   # spavaj 2 sekunde da mogu da procitam sta je napisano

# sad kad imamo koeficijente, hajmo da prikazemo kalibrisanu sliku
# isto cemo raditi realtime sa kamerom

cv2.destroyAllWindows()  # ovo je da bi se ugasio onaj jedan prozor koji pokazuje uocene tacke

# Stereo kalibrisani preview

ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

dst = 0  # ovo ce biti slike koje nisu distortovane od cap0 i cap1 respektivno
dst1 = 0

cameraCompensation = 0.15

def getFrameCamera0():
	global frame0
	global newcameramtx
	global roi
	global mtx
	global dist
	global imgShape
	global dst
	global cameraCompensation

	time.sleep(cameraCompensation)    # kompenzacija za sinhronizaciju (valjda je dovoljna)
	ret0, frame0 = cap0.read()

	# refine image
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, imgShape, 1, imgShape)

	# undistort the image
	dst = cv2.undistort(frame0, mtx, dist, None, newcameramtx)  # dst ti je slika koja nije distortovana

	# nisam siguran da li da je croppujem jer zbog Depth map
	#x, y, w, h = roi
	#dst = dst[y:y+h, x:x+w]


def showFrameCamera0():
	global frame0
	cv2.imshow('leva', frame0)


def getFrameCamera1():
	global frame1
	global newcameramtx1
	global roi1
	global mtx1
	global imgShape1
	global dst1

	ret1, frame1 = cap1.read()

	#refine image
	newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, imgShape1, 1, imgShape1)

	#undistort the image
	dst1 = cv2.undistort(frame1, mtx1, dist1, None, newcameramtx1)  # dst1 je slika koja nije distortovana


def showFrameCamera1():
	global frame1
	cv2.imshow('desna', frame1)


# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (22, 71, 220)   #TODO vrednosti boje
greenUpper = (50, 194, 255)
pts = deque(maxlen=64)

stereo = cv2.StereoBM_create(64, 45)    # inicijalizacija depth map-e

getFrameCamera0()   # sluzi da bismo inicijalizovali nizove kako bi mogli da rade asinhrono kasnije
getFrameCamera1()

kompenzacijaDMx = 0
kompenzacijaDMy = 0

x = 0
y = 0
radius = 0

consecutiveCenters = 0  # sluzi za iscrtavanje predikcije putanje
korak = 5  # korak iteracije u plotovanju predikcije putanje

while True:
	start = timeit.default_timer()
	# preuzmi sliku, rafiniraj je i ukloni distorziju

	g0 = Thread(target=getFrameCamera0)
	g1 = Thread(target=getFrameCamera1)

	g0.start()
	g1.start()

	#g0.join()
	#g1.join()

	frame0calib = np.copy(dst)
	frame1calib = np.copy(dst1)
	cv2.circle(frame0calib, (320, 240), 10, (255, 0, 0), 3)
	cv2.circle(frame1calib, (320, 240), 10, (255, 0, 0), 3)

	cv2.imshow('leva', frame0calib)  # pirkazujemo popravljenu sliku
	cv2.imshow('desna', frame1calib)

	cv2.imshow('dst', dst)
	cv2.imshow('dst1', dst1)

	keypressed = cv2.waitKey(1) & 0xFF
	if keypressed == ord('p'):   # mogucnost zamrzavanja slike na slovo 'p'
		i = 1
		while True:
			thisWhileKeyPressed = cv2.waitKey(1) & 0xFF
			if thisWhileKeyPressed == ord('c'):
				break
			if thisWhileKeyPressed == ord('0'):
				cameraCompensation += 0.001
			if thisWhileKeyPressed == ord('9'):
				cameraCompensation -= 0.001
			if thisWhileKeyPressed == ord('-'):
				cameraCompensation += 0.01
			if thisWhileKeyPressed == ord('='):
				cameraCompensation += 0.1
			if thisWhileKeyPressed == ord('8'):
				cameraCompensation -= 0.01
			if thisWhileKeyPressed == ord('7'):
				cameraCompensation -= 0.1
			print 'camera compensation = ' + str(cameraCompensation)
			i += 1

	# TRACKING THE BALL
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(dst, width=600)  # radicemo sa levom slikom # Gasimo ovo zbog pracenja na DM slici...
	# frame = np.copy(dst)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # ovo se radi da bi se pokusali eliminisati sumovi
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# ok, sad smo sve maskirali sta nam treba, treba naci loptu i iscrtati je

	# find contours in the mask and initialize the current (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)  # u slucaju da je nadjeno vise od jedne konture, uzimamo najvecu
		((x, y), radius) = cv2.minEnclosingCircle(c)  # aproksimiramo najmanju obuhvatajucu konturu i uzimamo joj centar
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 15:  #TODO podesavanje najmanjeg poluprecnika
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	if center is None:
		consecutiveCenters = 0
	else:
		if consecutiveCenters < 5:  # potreban nam je podatak kada se naredjalo 5 uzastopnih centara
			consecutiveCenters += 1

	# posto smo nasli centar i dodali ga na listu, crtamo trag

	# loop over the set of tracked points
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)   # 64 ti je broj tacaka koliko se iscrtava u tragu
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# posto mi vise definitivno ne trebaju originalne popravljene slike, mogu direktno sa njima da radim
	#TODO proracun DepthMap-e
	depthMapFrame0 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	depthMapFrame1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)

	disparity = stereo.compute(depthMapFrame0, depthMapFrame1, cv2.CV_32F)

	res = cv2.convertScaleAbs(disparity)    # ovo je da bismo mogli da prikazemo sliku u realnom vremenu za razliku od matplotlib

	res = imutils.resize(res, width=600)    # smanjujemo sliku na 600 da bi sistem brze radio

	resWithCircle = np.copy(res)
	cv2.circle(resWithCircle, (int(x+kompenzacijaDMx), int(y+kompenzacijaDMy)), int(radius), (127, 127, 127), 2)    # neka bude sive boje
	cv2.circle(resWithCircle, (int(x+kompenzacijaDMx), int(y+kompenzacijaDMy)), 5, (127, 127, 127), -1)

	cv2.imshow('disparity', resWithCircle)

	# show the frame to our screen
	frameWithNums = np.copy(frame)
	#cv2.putText(frameWithNums, str(center), (0, 15), cv2.QT_FONT_NORMAL, 0.5, (0, 255, 0), 1)    # todo zavrsi ispis koordinata, za sad je samo 2D

	if center is not None:
		# TODO ovde ti ode van indexa u DM posto je ona manja
		#cv2.putText(frameWithNums, '(' + str(center[0]) + ', ' + str(center[1]) + ', ' + str(
		#	res[int(x+kompenzacijaDMx), int(y+kompenzacijaDMy)]) + ')', (0, 15), cv2.QT_FONT_NORMAL, 0.5, (0, 255, 0), 1)
		cv2.putText(frameWithNums, str(center), (0, 15), cv2.QT_FONT_NORMAL, 0.5, (0, 255, 0), 1)
		cv2.putText(frameWithNums, '(' + str(res[center[1], center[0]]) + ')', (0, 30), cv2.QT_FONT_NORMAL, 0.5, (0, 255, 0), 1)

	# pre iscrtavanja slike, zelimo jos da dodamo i predikciju putanje na sliku
	if consecutiveCenters == 5:  # ukoliko je primeceno 5 ili vise uzastopnih centara, racunaj putanju u odnosu na 1., 3. i 5. tacku
		(x1, y1) = (float(pts[0][0]), float(pts[0][1]))
		(x2, y2) = (float(pts[2][0]), float(pts[2][1]))
		(x3, y3) = (float(pts[4][0]), float(pts[4][1]))

		# zakrpa da denom ne bude 0
		if (x1 == x2) | (x1 == x3) | (x2 == x3):
			denom = 0.000000000001
		else:
			denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

		A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
		B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
		C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

		h, w = frame.shape[:2]

		x = int(x1)
		y = int(A * x1**2 + B * x1 + C)

		calculatedPoints = deque()
		calculatedPoints.clear()    # sada nije potrebno, ali ce trebati kasnije

		if x1 > x3:
			while (0 < y < h) & (0 < x < w):
				y = int(A * x**2 + B * x + C)
				calculatedPoints.appendleft((x, y))
				x += korak
		else:
			while (0 < y < h) & (0 < x < w):
				y = int(A * x**2 + B * x + C)
				calculatedPoints.appendleft((x, y))
				x -= korak


		# sada smo sracunali sve tacke funkcije i trebamo ih iscrtati
		for i in xrange(1, len(calculatedPoints)):
			if calculatedPoints[i - 1] is None or calculatedPoints[i] is None:
				continue    # trebaju nam makar 2 tacke da bismo mogli da iscrtamo liniju
			cv2.line(frameWithNums, calculatedPoints[i - 1], calculatedPoints[i], (0, 255, 0), 5)

	cv2.imshow("Frame", frameWithNums)

	cv2.waitKey(1)

	if keypressed == ord('q'):
		break

	# w, s, a, d za kalibraciju DM
	if keypressed == ord('d'):
		kompenzacijaDMx += 1

	if keypressed == ord('a'):
		kompenzacijaDMx -= 1

	if keypressed == ord('w'):
		kompenzacijaDMy -= 1

	if keypressed == ord('s'):
		kompenzacijaDMy += 1

	# posto nisam siguran kako da sinhronizujem kamere kako treba, da bismo koliko toliko sprecili
	# zauzetost resursa u isto vreme, ogranicimo kamere na <30 fps
	#time.sleep(0.035)
	stop = timeit.default_timer()

	print 'iteracija je trajala: ' + str(stop - start)


cap0.release()
cap1.release()
cv2.destroyAllWindows()

# dovde
