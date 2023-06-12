import cv2
import cvzone
import time
from keypoints import getKeypoints


class captureImage:

    def __init__(self):
        self.timer = 5
        self.textMessage = 'Default Message'
        self.distanceMessages = ['Move closer', 'Stay there', 'Move farther']
        self.standingTimer = 0
        self.pixels_width = self.angleDistance = 0
        self.angle = 0
        # [male eye length, female eye length]
        self.real_width = [2.755, 2.67]
        self.focal_length = 510
        # self.distanceInterval = [180.00, 200.00]
        self.distanceInterval = [40.00, 50.00]
        self.distances = []

    def capture(self, gender=None, distanceInterval=None):
        if gender == 'Male':
            self.real_width = self.real_width[0]
        elif gender == 'Female':
            self.real_width = self.real_width[1]
        else:
            self.real_width = sum(self.real_width)/len(self.real_width)

        cap = cv2.VideoCapture(0)
        keypointsObj = getKeypoints()

        while True:
            ret, img = cap.read()
            landmarks = keypointsObj.getKeypoints(img)
            img = keypointsObj.drawKeypoints(img, landmarks)
            k = cv2.waitKey(1)
            if k == 27:
                cap.release()
                return -1
            try:
                if landmarks:
                    lefteyeInner = landmarks[1]
                    lefteyeOuter = landmarks[3]
                    righteyeInner = landmarks[4]
                    righteyeOuter = landmarks[6]
                    lefteyeDistance = keypointsObj.getDistance(
                        lefteyeInner, lefteyeOuter)
                    righteyeDistance = keypointsObj.getDistance(
                        righteyeInner, righteyeOuter)

                    eyeVertical = (
                        lefteyeInner[2]+lefteyeOuter[2]+righteyeInner[2]+righteyeOuter[2])/4

                    # w = width in pixels
                    # W = width in cm
                    self.pixels_width = (lefteyeDistance + righteyeDistance)/2

                    # finding distance between camera and person
                    self.angleDistance = (self.real_width *
                                          self.focal_length)/self.pixels_width

                    if distanceInterval == 1:
                        self.distanceInterval = [150.00, 220.00]

                    if round(self.angleDistance, 2) < self.distanceInterval[0]:
                        self.textMessage = self.distanceMessages[2]
                        self.standingTimer = 0
                    elif round(self.angleDistance, 2) > self.distanceInterval[1]:
                        self.textMessage = self.distanceMessages[0]
                        self.standingTimer = 0
                    else:
                        self.textMessage = self.distanceMessages[1]
                    cvzone.putTextRect(
                        img, f'Distance: {round(self.angleDistance, 2)}cm', (40, 40), scale=2)
                    cvzone.putTextRect(
                        img, self.textMessage, (250, 450), scale=2)
                    self.distances.append(self.angleDistance)

                    cv2.namedWindow("Image")
                    cv2.moveWindow("Image", 50, 50)
                    cv2.imshow("Image", img)
                    cv2.waitKey(1)

                    if self.textMessage == self.distanceMessages[1]:
                        self.standingTimer += 1
                        cv2.waitKey(1)

                    if self.standingTimer == 50:
                        cv2.destroyWindow("Image")
                        while True:
                            prev = time.time()
                            while self.timer >= 0:
                                ret, img1 = cap.read()
                                cur = time.time()
                                cvzone.putTextRect(
                                    img1, "Capturing picture in "+str(self.timer), (50, 450), scale=2)
                                cv2.namedWindow("Capture")
                                cv2.moveWindow("Capture", 50, 50)
                                cv2.imshow("Capture", img1)
                                cv2.waitKey(1)
                                if cur-prev >= 1:
                                    prev = cur
                                    self.timer -= 1
                            else:
                                ret, img1 = cap.read()

                                if self.distanceInterval[0] == 40.00:
                                    cv2.imwrite(
                                        'results/captured-close.jpg', img1)
                                    print(
                                        "Captured image and saved as 'captured-close.jpg'.")
                                    cap.release()
                                    cv2.destroyAllWindows()
                                else:
                                    cv2.imwrite(
                                        'results/captured-body.jpg', img1)
                                    print(
                                        "Captured image and saved as 'captured-body.jpg'.")
                                    cap.release()
                                    cv2.destroyAllWindows()
                                # self.angleDistance = round(sum(self.distances)/len(self.distances), 2)
                                return [self.focal_length, self.angleDistance, img1, eyeVertical]
            except:
                pass
            cv2.imshow("Image", img)
