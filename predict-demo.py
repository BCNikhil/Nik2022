from agp import PredictAgeGender
from captureImage import captureImage
from keypoints import getKeypoints
from findDistance import Distance
from knn import KNNClassifier
import cv2

captureCloseObj = captureImage()

# imageClose = captureCloseObj.capture()
captureClose = cv2.imread('results/captured-close.jpg')
print(captureClose.shape)
cv2.imshow("close", captureClose)
cv2.waitKey(1)
cv2.destroyAllWindows()
focal = 510
# focal, distanceClose, captureClose = imageClose[0], imageClose[1], imageClose[2]
if type(captureClose) == int:
    print("Exited. Could not get close picture.")
else:
    agp = PredictAgeGender()
    agpRes = agp.getResults(captureClose)
    gender, age = agpRes[0], agpRes[1]
    captureBodyObj = captureImage()
    # imageBody = captureBodyObj.capture(
    # gender=gender, distanceInterval=1)
    # focal, distanceBody, captureBody = imageBody[0], imageBody[1], imageBody[2]
    # print(distanceBody)
    captureBody = cv2.imread('results/captured-body.jpg')
    cv2.imshow("body", captureBody)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    if type(captureBody) == int:
        print("Exited. Could not get full body picture.")
    else:
        keypointsObj = getKeypoints()
        keypoints = keypointsObj.getKeypoints(captureBody)
        # if len(keypoints) < 30:
        #     print("Exited. \nReason: The full body of the person was not captured.")
        #     exit(0)
        keypointsImage = keypointsObj.drawKeypoints(captureBody, keypoints)
        distance = Distance(keypoints, focal, 199.213)
        collar = distance.findCollar()
        print("Collar dist in cm:", collar)
        chest = distance.findChest()
        print("Chest dist in cm:", chest)
        waist = distance.findWaist()
        print("Waist dist in cm:", waist)
        hip = distance.findHip()
        print("Hip dist in cm:", hip)
        thigh = distance.findThigh()
        print("Thigh dist in cm:", thigh)
        # inseam = distance.findInseam()
        # print("Inseam dist in cm:", inseam)

        knnModel = KNNClassifier()
        res = knnModel.knnPredict([gender, age, chest, waist, hip, collar, thigh])
        print("Predicted size:",res)