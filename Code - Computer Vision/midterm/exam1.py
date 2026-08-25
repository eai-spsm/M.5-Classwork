# Midterm scratch: batch Haar face-detect over faces/img_NNN.jpg -> output/imgN.jpg.
# Entirely commented out (early draft) -- see exam2.py for the working version.
# import cv2 as cv
# import os

# xml_path = "C:\\Users\\SPSM LAB 7\\Documents\\024329\\computer_vision-main\\haarcascade_frontalface_default.xml"

# detected = 0

# for i in range (1,121):
# 	if i < 10:
# 		img = cv.imread(f"C:\\Users\\SPSM LAB 7\\Documents\\024329\\computer_vision-main\\faces\\img_00{i}.jpg")
# 	elif i <100:
# 		img = cv.imread(f"C:\\Users\\SPSM LAB 7\\Documents\\024329\\computer_vision-main\\faces\\img_0{i}.jpg")
# 	else:
# 		img = cv.imread(f"C:\\Users\\SPSM LAB 7\\Documents\\024329\\computer_vision-main\\faces\\img_{i}.jpg")
# 	face_cascde = cv.CascadeClassifier(xml_path)
# 	gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# 	scale = 1.9
# 	minNeighbor = 1
# 	face_detect = face_cascde.detectMultiScale(gray_img, scale, minNeighbor)
# 	if len(face_detect) > 0:
#     		for x, y, w, h in face_detect:
#        			cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     		cv.imwrite(f"C:\\Users\\SPSM LAB 7\\Documents\\024329\\computer_vision-main\\output\\img{i}.jpg", img)
# 	else:
#     	    	print(f"No faces detected. {i}")
# 	num = len(face_detect)
# 	detected += num

# print(f"detected: {detected} faces")
