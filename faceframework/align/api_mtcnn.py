import numpy as np
import torch
from PIL import Image,ImageDraw
#from align.mtcnn import detector
from torch.autograd import Variable
from faceframework.align.mtcnn.get_nets import PNet, RNet, ONet
from faceframework.align.mtcnn.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from faceframework.align.mtcnn.first_stage import run_first_stage
from faceframework.align.mtcnn.align_trans import get_reference_facial_points, warp_and_crop_face
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class MTCNN():
    def __init__(self):
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square= True)
        
    def align(self, img):
        boxes, landmarks = self.detect_faces(img)
        facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        return boxes,[Image.fromarray(warped_face)]
    
    def align_multi(self, img, limit=None, min_face_size=30.0):
        if type(img) is np.ndarray:
            img = Image.fromarray(img[...,::-1])

        boxes, landmarks = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]

        for i in range(len(bboxes)):
            facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]

            facial5points_tuple=[(landmarks[i][j],landmarks[i][j+5]) for j in range(5)]
            draw.polygon(facial5points_tuple,fill= (255, 0, 0))

            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            dstimage_name = '.'.join(image_name.split('.')[:-1]) + '_aligned_%d.jpg'%i
            img_warped.save(os.path.join(dest_root, subfolder, dstimage_name))

        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def detect_faces(self, image, min_face_size=20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.8, 0.7, 0.7]):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """


        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            if len(bounding_boxes)==0:
                return [],[]
            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes)==0:
                # 前面为[None,None,...,None]
                return [],[]
            bounding_boxes = np.vstack(bounding_boxes)

            #zjhtest
            picnew=image.copy()
            draw = ImageDraw.Draw(picnew)
            for b in bounding_boxes:
                draw.rectangle([
                    (b[0], b[1]), (b[2], b[3])
                ], outline = 'white')
            picnew.save('data/people/stage1_'+time.asctime().replace(' ','').replace(':','')+'_befor_nms.jpg')

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            #zjhtest
            picnew=image.copy()
            draw = ImageDraw.Draw(picnew)
            for b in bounding_boxes:
                draw.rectangle([
                    (b[0], b[1]), (b[2], b[3])
                ], outline = 'orange')
            picnew.save('data/people/stage1_'+time.asctime().replace(' ','').replace(':','')+'.jpg')
            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            #zjhtest
            picnew=image.copy()
            draw = ImageDraw.Draw(picnew)
            for b in bounding_boxes:
                draw.rectangle([
                    (b[0], b[1]), (b[2], b[3])
                ], outline = 'yellow')
            picnew.save('data/people/stage2_'+time.asctime().replace(' ','').replace(':','')+'.jpg')
            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0: 
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            #zjhtest
            picnew=image.copy()
            draw = ImageDraw.Draw(picnew)
            for b in bounding_boxes:
                draw.rectangle([
                    (b[0], b[1]), (b[2], b[3])
                ], outline = 'green')
            picnew.save('data/people/stage3_'+time.asctime().replace(' ','').replace(':','')+'_beforenms.jpg')

            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]

            #zjhtest
            picnew=image.copy()
            draw = ImageDraw.Draw(picnew)
            for b in bounding_boxes:
                draw.rectangle([
                    (b[0], b[1]), (b[2], b[3])
                ], outline = 'red')
            picnew.save('data/people/stage3_'+time.asctime().replace(' ','').replace(':','')+'_nms.jpg')

            landmarks = landmarks[keep]

        return bounding_boxes, landmarks



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