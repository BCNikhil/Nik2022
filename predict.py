from agp import PredictAgeGender
from captureImage import captureImage
from keypoints import getKeypoints
from findDistance import Distance
from knn import KNNClassifier
from math import asin, cos
import pandas as pd

# angle of eyeline from camera, horizontal distance from camera, keypoint variable, 
angle, realDistance, keypointNotFound = 0, 0, 0

try:
    captureCloseObj = captureImage()
except:
    print("Exited. \nReason: Could not create captureImage object.")
    exit(0)

try:
    imageClose = captureCloseObj.capture()
    focal, distanceClose, captureClose, eyeVerticalClose = imageClose[
        0], imageClose[1], imageClose[2], imageClose[3]
except:
    print("Exited. Could not get close picture.")
    exit(0)


try:
    agp = PredictAgeGender()
    agpRes = agp.getResults(captureClose)
    gender, age = agpRes[0], agpRes[1]
except:
    print("Exited. \nReason: Could not create PredictAgeGender object.")
    exit(0)

try:
    captureBodyObj = captureImage()
except:
    print("Exited. \nReason: Could not create captureImage object.")
    exit(0)

try:
    imageBody = captureBodyObj.capture(
        gender=gender, distanceInterval=1)
    focal, distanceBody, captureBody, eyeVerticalBody = imageBody[
        0], imageBody[1], imageBody[2], imageBody[3]
    angle = asin(eyeVerticalBody/distanceBody)
    # print("Angle of eye from camera:", round(angle*180*7/22, 2))
    realDistance = round(distanceBody * cos(angle), 3)
    print("Distance from camera:", realDistance)
except:
    print("Exited. Could not get full body picture.")
    exit(0)

try:
    keypointsObj = getKeypoints()
    keypoints = keypointsObj.getKeypoints(captureBody)
    keypointsImage = keypointsObj.drawKeypoints(captureBody, keypoints)
except:
    print("Exited. \nReason: Could not create getKeypoints object.")
    exit(0)

try:    
    distance = Distance(keypoints, focal, realDistance)
except:
    print("Exited. \nReason: Could not create Distance object.")
    exit(0)

try:
    collar = distance.findCollar()
    print("Collar dist:", f"{collar}cm")
except:
    keypointNotFound = 1
    collar = None
    print("Full body image not taken properly.")

try:
    chest = distance.findChest()
    print("Chest dist:", f"{chest}cm")
except:
    keypointNotFound = 1
    chest = None
    print("Full body image not taken properly.")

try:
    waist = distance.findWaist()
    print("Waist dist:", f"{waist}cm")
except:
    keypointNotFound = 1
    waist = None
    print("Full body image not taken properly.")

try:
    hip = distance.findHip()
    print("Hip dist:", f"{hip}cm")
except:
    keypointNotFound = 1
    hip = None
    print("Full body image not taken properly.")

try:
    sleeveHalf = distance.findHalfSleeve()
    print("Half sleeve dist:", f"{sleeveHalf}cm")
except:
    # keypointNotFound = 1
    sleeveHalf = None
    print("Full body image not taken properly. (Half sleeve)")

try:
    sleeveFull = distance.findFullSleeve()
    print("Full sleeve dist:", f"{sleeveFull}cm")
except:
    # keypointNotFound = 1
    sleeveFull = None
    print("Full body image not taken properly. (Full Sleeve)")

try:
    inseam = distance.findInseam()
    print("Inseam dist:", f"{inseam}cm")
except:
    keypointNotFound = 1
    inseam = None
    print("Full body image not taken properly.")

try:
    thigh = distance.findThigh()
    print("Thigh dist:", f"{thigh}cm")
except:
    keypointNotFound = 1
    thigh = None
    print("Full body image not taken properly.")

if keypointNotFound == 1:
    print("Exited. \nReason: Could not calculate enough number of measurements.")
    exit(0)
else:
    try:
        model = KNNClassifier()
        dataDict = {'gender': [gender], 'age': [age], 'chest': [chest], 'waist': [waist], 'hip': [
            hip], 'half-sleeve': [sleeveHalf], 'full-sleeve': [sleeveFull], 'collar': [collar], 'thigh': [thigh]}
        df = pd.DataFrame(dataDict)
        res = model.knnPredict(
            [gender, age, chest, waist, hip, collar, thigh])
        print("Predicted Size:", res[0])
        print(df)
    except:
        print("Exited. \nReason: Could not create KNNClassifier object.")
        exit(0)