#File used to convert video to image sequence for annotation

import cv2

# Opens the Video file
cap= cv2.VideoCapture('pitcher_vids/cut up/cutup.mp4')
i=0
# Loops through each frame, writing each one to a file
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('sequence/'+str(i)+'.jpg',frame)
    i+=1
# Releases and closes the video file
cap.release()
cv2.destroyAllWindows()