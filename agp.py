import cv2


class PredictAgeGender:

    def __init__(self):
        self.padding = 20

        self.faceProto = "assets/opencv_face_detector.pbtxt"
        self.faceModel = "assets/opencv_face_detector_uint8.pb"

        self.ageProto = "assets/age_deploy.prototxt"
        self.ageModel = "assets/age_net.caffemodel"

        self.genderProto = "assets/gender_deploy.prototxt"
        self.genderModel = "assets/gender_net.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                        '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']

        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

    def getFaceBox(self, net, frame, confThreshold=0.7):
        frameOpenCVDNN = frame.copy()
        frameHeight = frameOpenCVDNN.shape[0]
        frameWidth = frameOpenCVDNN.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpenCVDNN, 1.0, (300, 300), [
                                    104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpenCVDNN, (x1, y1), (x2, y2),
                             (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpenCVDNN, boxes

    def getResults(self, img):

        frameFace, boxes = self.getFaceBox(self.faceNet, img)
        if not boxes:
            print("No face detected.")
            return -1
        for bbox in boxes:
            # print(bbox)
            face = img[max(0, bbox[1]-self.padding):min(bbox[3]+self.padding, img.shape[0]-1),
                       max(0, bbox[0]-self.padding):min(bbox[2]+self.padding, img.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]
            #print("Age Output : {}".format(agePreds))

            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            if age in self.ageList[:3]:
                age = 1
            else:
                age = 0

            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = self.genderList[genderPreds[0].argmax()]
            #print("Gender Output : {}".format(genderPreds))

            if age == 1:
                print("Child")
            else:
                print("Gender : {}, conf = {:.3f}".format(
                    gender, genderPreds[0].max()))

            if age == 1:
                gender = 2
            elif gender == 'Male':
                gender = 0
            elif gender == 'Female':
                gender = 1

        return [gender, age]
