from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch,numpy as np
import sys,os,time
from PIL import Image
sys.path.append('..')
from faceframework.cnn_impl.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from faceframework.cnn_impl.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
#from faceframework.4_app.extract_feature.extract_feature_v1 import extract_feature
from faceframework.config import get_config
from faceframework.align.api_mtcnn import MTCNN
from faceframework.align.api_centerface import CenterFaceAPI
from faceframework.utils import featureutils





class FaceModel:
    '''
    zhujinhua@20200602
    Face detect，align and extract feature
    '''
    def __init__(self, args):
        self.args = args
        self.conf = get_config(False)

        self.model = self.loadmodel('faceframework'/self.conf.model_path/self.conf.model_name)

        mtcnn = MTCNN()
        centerface = CenterFaceAPI()
        print('centerface loaded')
        self.detector = mtcnn


    def loadmodel(self, model_root, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        '''
        zhujinhua 20200531
        :param model_root: model file save path
        :param device:
        :return:
        '''
        assert (os.path.exists(model_root))
        print('Backbone Model Root:', model_root)
        INPUT_SIZE=[112,112]
        BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                         'ResNet_101': ResNet_101(INPUT_SIZE),
                         'ResNet_152': ResNet_152(INPUT_SIZE),
                         'IR_50': IR_50(INPUT_SIZE),
                         'IR_101': IR_101(INPUT_SIZE),
                         'IR_152': IR_152(INPUT_SIZE),
                         'IR_SE_50': IR_SE_50(INPUT_SIZE),
                         'IR_SE_101': IR_SE_101(INPUT_SIZE),
                         'IR_SE_152': IR_SE_152(INPUT_SIZE)}
        BACKBONE_NAME='IR_50'
        backbone = BACKBONE_DICT[BACKBONE_NAME]

        # load backbone from a checkpoint
        print("Loading Backbone Checkpoint '{}'".format(model_root))
        if device.type=='cpu':
            backbone.load_state_dict(torch.load(model_root,map_location='cpu'))
        else:
            backbone.load_state_dict(torch.load(model_root))
        backbone.to(device)
        backbone.eval() # set to evaluation mode
        return backbone
    def l2_norm(self, input, axis = 1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output


    def get_imgface(self,img):
        """
        zhujinhua 20200602
        Arguments:
            img: an instance of PIL.Image.
            min_face_size: a float number.
        Returns:
            faces:
            bboxes:
            facecount
        """
        start=time.time()
        bboxes, faces = self.detector.align_multi(img, self.conf.face_limit, self.conf.min_face_size)
        if len(bboxes) == 0:
            return [],[],0
        bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(int)
        offsetpixes = 20
        bboxes = bboxes + [-offsetpixes,-offsetpixes,offsetpixes,offsetpixes] # personal choice
        bboxes = [[x[0] if x[0]>=0 else 0,x[1] if x[1]>=0 else 0,x[2],x[3]] for x in bboxes]
        print(time.time()-start)
        return faces,bboxes,len(bboxes)


    def get_feature(self, aligned):
        '''
        zhujinhua 20200531
        :param aligned: an aligned face, a PIL.Image instance or np.ndarray
        :return: feature
        '''
        embedding_size = 512
        #features = np.zeros([len(aligned), embedding_size])
        with torch.no_grad():
            input = self.conf.test_transform(aligned).to(self.conf.device)
            features = self.model(input.unsqueeze(0))
            features = self.l2_norm(features).cpu()
            #self.model(aligned)
            return features.numpy()

    # zhujinhua@20200531
    def get_batchfeature(self, batchaligned):
        '''
        zhujinhua@20200531
        :param batchaligned: batch aligned face, a list of PIL.Image instance
        :return: features
        '''
        embedding_size = 512
        if len(batchaligned)==0:
            return []
        with torch.no_grad():
            #features = np.zeros([len(aligned), embedding_size])
            data = [self.conf.test_transform(face).numpy() for face in batchaligned]
            input = torch.tensor(data).to(self.conf.device)

            features = self.model(input)
            features = self.l2_norm(features).cpu()
            return features.numpy()

    def facecompare(self,img1,img2):
        result={'result':'success','similarity':0}
        faces1, bb_box, face_num = self.get_imgface(img1)
        if faces1 == None or len(faces1)==0:
            result['result']='fail'
            result['msg'] ='img1 detect no face'
            return result

        # batch get all features
        features1 = self.get_batchfeature(faces1)
        faces2, bb_box, face_num = self.get_imgface(img2)
        if faces2 == None or len(faces2)==0:
            result['result']='fail'
            result['msg'] ='img2 detect no face'
            return result

        # batch get all features
        features2 = self.get_batchfeature(faces2)
        sim=np.dot(features1[0],features2[0])
        result['similarity']=str(featureutils.similaritymap(sim))
        return result

    def facesearch(self,facelib,img1):
        result={'result':'success','similarity':0}
        faces1, bb_box, face_num = self.get_imgface(img1)
        if faces1 == None or len(faces1)==0:
            result['result']='img1 detect no face'
            return result

        # batch get all features
        features1 = self.get_batchfeature(faces1)

        topk = facelib.feature_comparetopk(features1,20)
        topk['cossimilarity']=['%.3f'%featureutils.similaritymap(sim) for sim in topk['cossimilarity']]
        return topk

import cv2
if __name__ == '__main__':
    global facemodel
    facemodel = FaceModel('')
    frame = Image.open("../data/imgs/1.jpg")
    t1=time.time()
    faces, bboxes, facecont = facemodel.get_imgface(frame)
    t2=time.time()
    print(t2-t1)
    for face in faces:
        f=facemodel.get_feature(face)
        print(f)

    #f=facemodel.get_feature(np.array([np.asarray(x) for x in faces]).transpose((0,3,1,2)))
    f=facemodel.get_batchfeature(faces)
    print(f)
    image = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
    for idx,bbox in enumerate(bboxes):
        image = facemodel.detector.draw_box_name(bbox, 'xx', image)
    cv2.imshow('face Capture', image)
    cv2.waitKey(0)&0xFF == ord('q')


    #print(bboxes)
    #print(faces)

