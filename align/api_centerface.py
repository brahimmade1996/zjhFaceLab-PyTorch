import numpy as np
import torch
from PIL import Image
#from align.mtcnn import detector
from torch.autograd import Variable
from faceframework.align.centerface.centerface import CenterFace
from faceframework.align.mtcnn.align_trans import get_reference_facial_points, warp_and_crop_face

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class CenterFaceAPI():
    def __init__(self):
        self.landmarks = True
        self.centerface = CenterFace(landmarks=self.landmarks)
        self.refrence = get_reference_facial_points(default_square= True)

    
    def align_multi(self, img, limit=None, min_face_size=30.0):
        if type(img) is np.ndarray:
            frame = img
        else:
            frame = np.array(img)
        h, w = frame.shape[:2]

        boxes, landmarks = self.centerface(frame, h, w, threshold=0.55)


        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[2*j],landmark[2*j+1]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
            faces.append(Image.fromarray(warped_face[...,::-1]))
        return boxes, faces


    def draw_box_name(self,bbox,name,frame):
        frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
        frame = cv2.putText(frame,
                            name,
                            (bbox[0],bbox[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0,255,0),
                            3,
                            cv2.LINE_AA)
        return frame

import cv2,time
if __name__ == '__main__':
    mtcnn=MTCNN()
    image = cv2.imread("..\\data\\imgs\\1.jpg")
    frame = Image.fromarray(image[...,::-1])
    frame = Image.open("../../data/imgs/1.jpg")
    t1=time.time()
    bboxes, faces = mtcnn.align_multi(frame)
    t2=time.time()
    print(t2-t1)

    #print(bboxes)
    #print(faces)
    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
    bboxes = bboxes.astype(int)
    bboxes = bboxes + [-1,-1,1,1] # personal choice
    #results, score = learner.infer(conf, faces, targets, args.tta)
    for idx,bbox in enumerate(bboxes):
       image = mtcnn.draw_box_name(bbox, 'xx', image)
    cv2.imshow('face Capture', image)
    cv2.waitKey(0)&0xFF == ord('q')