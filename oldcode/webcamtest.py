import cv2
webcam = cv2.VideoCapture(0)
check, frame = webcam.read()
print(check) #prints true as long as the webcam is running
#print(frame) #prints matrix values of each framecd 
cv2.imshow("Capturing", frame)
#Saving image in test folder
file = '/home/mikhail/Desktop/Robotics/oldcode/saved_img.jpg'
cv2.imwrite(filename=file, img=frame)
#cv2.waitKey(1650)
cv2.destroyAllWindows()
print("Processing image...")
img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
img_ = cv2.resize(img_,(28,28)) #Converts image size to 28x28
img_resized = cv2.imwrite(filename=file, img=img_)
print("Image saved!")
webcam.release()
#/home/mikhail/Desktop/Robotics/oldcode/webcamtest.py