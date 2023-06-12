import mediapipe as mp
import cv2
from math import sqrt


class getKeypoints:

    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.pose_landmarks = None

    def getKeypoints(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        landmarks = []
        if results.pose_landmarks:
            # self.mpDraw.draw_landmarks(
            #     img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            self.pose_landmarks = results.pose_landmarks
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                # height, width will be in interval [0.0, 1.0]
                height, width, channels = img.shape
                visibility = landmark.visibility
                if visibility > 0.4:
                    # get the keypoints with respect to image size
                    cx, cy = int(landmark.x*width), int(landmark.y*height)
                    landmarks.append([id, cx, cy, visibility])

        return landmarks

    def drawKeypoints(self, img, landmarks):
        for landmark in landmarks:
            cv2.circle(img, (landmark[1], landmark[2]),
                       5, (255, 0, 0), cv2.FILLED)
        self.mpDraw.draw_landmarks(
            img, self.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        # cv2.imwrite("results/keypoints-mediapipe.jpg", img)
        # print("Captured image and saved as 'keypoints-mediapipe.jpg'.")
        return img

    def getDistance(self, p1, p2):
        x1, y1, x2, y2 = p1[1], p1[2], p2[1], p2[2]
        distance = sqrt((x1-x2)**2+(y1-y2)**2)
        return distance
