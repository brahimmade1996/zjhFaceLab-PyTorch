import torch,numpy as np
import sys
sys.path.append('..')
from cnn_impl.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from cnn_impl.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from extract_feature.extract_feature_v1 import extract_feature


if __name__ == '__main__':
    img=''
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
    model_root='D:/OpenSource/FaceCNN/model/backbone_ir50_ms1m_epoch120.pth'
    #model_root='D:/OpenSource/CNNFace/model/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'



    #img_path='../data/ganalign/Seq1'
    img_path='../data/zperson/imgs'

    # use extract_feature_v1
    fs=extract_feature(img_path, backbone, model_root, batch_size=100,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = False)
    compareresult=fs.dot(fs.T)
    print(compareresult)
    print('img 2 cosine simility with other:',compareresult[2,:])

    print('img n cosine simility with n+1:')
    #rst=[]
    for i in range(len(compareresult)-1):
        #rst.append(compareresult[i,i+1])
        print(compareresult[i,i+1],', ',end='')
        if i%5==0:
            print(';')
    #print(np.array(rst).reshape([1,-1])[0,:])
