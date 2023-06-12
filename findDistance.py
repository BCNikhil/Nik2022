'''
Finding distances in pixels for the following measurements:
1. Shoulder
2. Collar
3. Inseam
4. Waist
5. Hip
6. Sleeve -  i. Half
            ii. Full
7. Chest
8. Thigh 
'''

from math import sqrt


class Distance:

    def __init__(self, keypoints, focal, distanceBody):
        self.collarDist = 0
        self.inseamDist = 0
        self.waistDist = 0
        self.hipDist = 0
        self.halfSleeveDist = 0
        self.fullSleeveDist = 0
        self.chestDist = 0
        self.thighDist = 0
        self.shoulderDist = 0
        self.keypoints = keypoints
        self.focal = focal
        self.distanceBody = distanceBody

    def findShoulder(self):
        leftShoulder = self.keypoints[11]
        rightShoulder = self.keypoints[12]
        shoulderDist = sqrt(
            (leftShoulder[1]-rightShoulder[1])**2+(leftShoulder[2]-rightShoulder)**2)

        # formaula for finding shoulder distance
        self.shoulderDist = shoulderDist

        return round(self.shoulderDist, 2)

    def findCollar(self):
        leftEar = self.keypoints[7]
        rightEar = self.keypoints[8]
        neckDistPixel = sqrt(
            (leftEar[1]-rightEar[1])**2+(leftEar[2]-rightEar[2])**2)
        neckDist = (self.distanceBody * neckDistPixel)/self.focal
        neckRadius = neckDist/2

        # formula for finding collar distance
        self.collarDist = round((2*(22/7)*neckRadius), 2)

        return round(self.collarDist, 2)

    def findInseam(self):
        leftHip = self.keypoints[23]
        rightHip = self.keypoints[24]
        leftHeel = self.keypoints[29]
        rightHeel = self.keypoints[30]
        rightInseamDist = sqrt(
            (rightHip[1]-rightHeel[1])**2+(rightHip[2]-rightHeel[2])**2)
        leftInseamDist = sqrt(
            (leftHip[1]-leftHeel[1])**2+(leftHip[2]-leftHeel[2])**2)

        inseamDistPixel = (rightInseamDist + leftInseamDist)/2
        inseamDist = (self.distanceBody * inseamDistPixel)/self.focal
        self.inseamDist = inseamDist

        return round(self.inseamDist, 2)

    def findWaist(self):
        leftShoulder = self.keypoints[11]
        rightShoulder = self.keypoints[12]
        leftHip = self.keypoints[23]
        rightHip = self.keypoints[24]

        # find formula to get keypoints of waist
        rightWaist = [
            rightShoulder[1]+(0.2*sqrt((rightShoulder[1]-rightHip[1])**2)), 1.72*rightShoulder[2]]
        leftWaist = [leftShoulder[1] -
                     (0.2*sqrt(leftShoulder[1]-leftHip[1])**2), 1.72*leftShoulder[2]]

        # try to use circle circumference formula to get the distance
        # find formula for waist distance
        waistDistPixel = sqrt(
            (rightWaist[0]-leftWaist[0])**2+(rightWaist[1]-leftWaist[1])**2)
        waistRadiusPixel = waistDistPixel/2
        waistRadius = (self.distanceBody * waistRadiusPixel)/self.focal
        waistDist = 2*(22/7)*sqrt((waistRadius**2+(waistRadius/2)**2)/2)
        self.waistDist = waistDist

        return round(self.waistDist, 2)

    def findHip(self):
        leftHip = self.keypoints[23]
        rightHip = self.keypoints[24]
        leftWrist = self.keypoints[15]
        rightWrist = self.keypoints[16]

        # formula to get real hip distance
        realLeftHip = [leftWrist[1]-0.2*sqrt((leftHip[1]-leftWrist[1])**2),
                       leftWrist[2]+sqrt((leftHip[2]-leftWrist[2])**2)/2]
        realRightHip = [rightWrist[1]+0.2*sqrt((rightHip[1]-rightWrist[1])**2),
                        rightWrist[2]+sqrt((rightHip[2]-rightWrist[2])**2)/2]
        hipDistPixel = sqrt(
            (realLeftHip[0]-realRightHip[0])**2+(realLeftHip[1]-realRightHip[1])**2)
        hipRadiusPixel = hipDistPixel/2
        hipRadius = (self.distanceBody * hipRadiusPixel)/self.focal
        hipDist = 2*(22/7)*sqrt((hipRadius**2+(hipRadius/2)**2)/2)

        # formula for hip distance
        self.hipDist = hipDist

        return round(self.hipDist, 2)

    def findHalfSleeve(self):
        leftShoulder = self.keypoints[11]
        rightShoulder = self.keypoints[12]
        leftElbow = self.keypoints[13]
        rightElbow = self.keypoints[14]
        rightHalfSleeveDist = sqrt(
            (rightShoulder[1]-rightElbow[1])**2+(rightShoulder[2]-rightElbow[2])**2)
        leftHalfSleeveDist = sqrt(
            (leftShoulder[1]-leftElbow[1])**2+(leftShoulder[2]-leftElbow[2])**2)

        halfSleeveDist = (rightHalfSleeveDist + leftHalfSleeveDist)/2
        self.halfSleeveDist = (halfSleeveDist*self.distanceBody)/self.focal

        return round(self.halfSleeveDist, 2)

    def findFullSleeve(self):
        leftElbow = self.keypoints[13]
        rightElbow = self.keypoints[14]
        leftWrist = self.keypoints[15]
        rightWrist = self.keypoints[16]
        leftShoulder = self.keypoints[11]
        rightShoulder = self.keypoints[12]
        rightSleeveDistBot = sqrt(
            (rightElbow[1]-rightWrist[1])**2+(rightElbow[2]-rightWrist[2])**2)
        leftSleeveDistBot = sqrt(
            (leftElbow[1]-leftWrist[1])**2+(leftElbow[2]-leftWrist[2])**2)
        rightSleeveDistTop = sqrt((rightElbow[1]-rightShoulder[1])**2+(rightElbow[2]-rightShoulder[2])**2)
        leftSleeveDistTop = sqrt((leftElbow[1]-leftShoulder[1])**2+(leftElbow[2]-leftShoulder[2])**2)
        rightSleeveDist = rightSleeveDistBot + rightSleeveDistTop
        leftSleeveDist = leftSleeveDistBot + leftSleeveDistTop

        fullSleeveDist = (rightSleeveDist + leftSleeveDist)/2

        self.fullSleeveDist = (self.distanceBody * fullSleeveDist)/self.focal

        return round(self.fullSleeveDist, 2)

    def findChest(self):
        leftShoulder = self.keypoints[11]
        rightShoulder = self.keypoints[12]
        leftHip = self.keypoints[23]
        rightHip = self.keypoints[24]

        # find formula to get chest keypoints
        # y = 0.25, x = 0.2
        leftChest = [leftShoulder[1]-0.2*(leftShoulder[1]-leftHip[1]),
                     leftShoulder[2]+(0.25*(leftHip[2]-leftShoulder[2]))]
        rightChest = [rightShoulder[1]+(0.2*(rightHip[1]-rightShoulder[1])),
                      rightShoulder[2]+(0.25*(rightHip[2]-rightShoulder[2]))]
        chestDistPixel = sqrt(
            (leftChest[0]-rightChest[0])**2+(leftChest[1]-rightChest[1])**2)
        chestRadiusPixel = chestDistPixel/2
        chestRadius = (self.distanceBody * chestRadiusPixel)/self.focal
        chestDist = 2*(22/7)*chestRadius

        # formula for chest distance
        self.chestDist = chestDist

        return round(self.chestDist, 2)

    def findThigh(self):
        leftShoulder = self.keypoints[11]
        rightShoulder = self.keypoints[12]
        leftMouth = self.keypoints[9]
        rightMouth = self.keypoints[10]

        rightThighDist = sqrt((rightShoulder[1]-rightMouth[1])**2)
        leftThighDist = sqrt((leftShoulder[1]-leftMouth[1])**2)
        thighDistPixel = (rightThighDist + leftThighDist)/2
        thighRadiusPixel = thighDistPixel/2
        thighRadius = (self.distanceBody * thighRadiusPixel)/self.focal

        thighDist = 2*(22/7)*sqrt(thighRadius**2)

        # formula for thigh distance
        self.thighDist = thighDist

        return round(self.thighDist, 2)
