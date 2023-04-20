import cv2
import numpy as np
import mediapipe as mp

class TrainingDataGeneratorANN:
    def __init__(
        self, 
        vid, 
        detection_confidence = 0.3, 
        tracking_confidence = 0.3, 
        complexity = 1
    ) -> None:
        self.__vid = vid
        self.__detection_confidence = detection_confidence
        self.__tracking_confidence = tracking_confidence
        self.__complexity = complexity
        self.__mp_pose = mp.solutions.pose


    def calculate_angle(self,a,b,c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
        angle = np.abs(radians *180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle

        return angle


    def generate_data(self):
        final_values_vector = []
        with self.__mp_pose.Pose(
            min_detection_confidence=self.__detection_confidence,
            min_tracking_confidence=self.__tracking_confidence,
            model_complexity=self.__complexity,
            smooth_landmarks = True ) as pose:
            
            if type(self.__vid) is str:
                self.__vid = cv2.VideoCapture("./training data/" + self.__vid)
            else:
                self.__vid = cv2.VideoCapture(self.__vid)
            
            while self.__vid.isOpened():
                success, image = self.__vid.read()

                if not success:
                    break
                    # TODO: Add smth here
                image = cv2.resize(image,(0,0),fx = 0.5, fy = 0.5)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
        
                image.flags.writeable = False
                results = pose.process(image)
                eyesVisible = False
                shoulderVisible = False
                try:
                    #code for pose extraction
                    landmarks = results.pose_landmarks.landmark
                    
                    #Check if both eyes are visible.
                    left_eye = [landmarks[self.__mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_EYE.value].y]
                    right_eye = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_EYE.value].y]
                    shoulder = [landmarks[self.__mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_r = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[self.__mp_pose.PoseLandmark.LEFT_ELBOW.value].x,  landmarks[self.__mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    elbow_r = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,  landmarks[self.__mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[self.__mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    wrist_r = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    nose = [landmarks[self.__mp_pose.PoseLandmark.NOSE.value].x,landmarks[self.__mp_pose.PoseLandmark.NOSE.value].y]
                    
                    #Get Tje Corridnates of Hip
                    left_hip = [landmarks[self.__mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_knee = [landmarks[self.__mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [landmarks[self.__mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    #Put the Values for visibility 
                    #visiblity for Eyes
                    landmarks[self.__mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0

                    #fOR NOSE
                    landmarks[self.__mp_pose.PoseLandmark.NOSE.value].visibility = 0
                    
                    landmarks[self.__mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0

                    #fOR eAR
                    landmarks[self.__mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0

                    #Check if both shoulders are visible.
                    left_ear = [landmarks[self.__mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[self.__mp_pose.PoseLandmark.LEFT_EAR.value].y]
                    right_ear = [landmarks[self.__mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[self.__mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                    #Midpointts
                    midpoint_shoulder_x = (int(shoulder[0] * image_width )+ int(shoulder_r[0] * image_width))/2
                    midpoint_shoulder_y = (int(shoulder[1] * image_height )+ int(shoulder_r[1] * image_height))/2

                    midpoint_hip_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
                    midpoint_hip_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2

                    based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x)/2)
                    based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y)/2)

                    neck_point_x = (int(nose[0] * image_width )+ int(midpoint_shoulder_x))/2
                    neck_point_y = (int(nose[1] * image_height) + int(midpoint_shoulder_y))/2

                    #angles 
                    left_arm_angle = int(self.calculate_angle(shoulder, elbow, wrist))
                    right_arm_angle = int(self.calculate_angle(shoulder_r, elbow_r, wrist_r))
                    left_leg_angle = int(self.calculate_angle(left_hip, left_knee, left_ankle))
                    right_leg_angle = int(self.calculate_angle(right_hip, right_knee, right_ankle))
                    left_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))

                    ppm = 10.8
                    left_arm_motion = (left_arm_angle* left_arm_length) / ppm
                    right_arm_motion = (right_arm_angle * left_arm_length) / ppm

                    mid_point_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
                    mid_point_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2

                    #cv2.circle(image,(int(mid_point_x) ,int(mid_point_y +30 )),15,(0,255,255),-1)

                    landmarks[self.__mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
                    landmarks[self.__mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0

                    final_values_vector.append((
                        left_arm_angle,
                        right_arm_angle,
                        left_leg_angle,
                        right_leg_angle,
                        left_arm_motion,
                        right_arm_motion,
                        ))
                except:
                    #print("bad smth")
                    pass
                    # TODO: figure out what here
                
                            #writing angles
                
                    # cv2.putText(image,"left elbow" + str(left_arm_angle),(int(image_width - 250),int(40)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
                    # cv2.putText(image,str(right_arm_angle),(int(elbow_r[0]* image_width  -40),int(elbow_r[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,244,244),2,cv2.LINE_AA)


                    # cv2.putText(image,str(left_leg_angle),(int(left_knee[0]* image_width + 40),int(left_knee[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
                    # cv2.putText(image,str(right_leg_angle),(int(right_knee[0]* image_width - 40),int(right_knee[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)

        self.__vid.release()
        cv2.destroyAllWindows()
        return final_values_vector


    def update_new_obj(self, filename: str) -> None:
        self.__vid = filename
