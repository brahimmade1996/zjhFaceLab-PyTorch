import torchvision.transforms as transforms
from datasetloader.data_pipe import load_bin, load_mx_rec
import argparse
from pathlib import Path
from config import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'for extracting faces_emore data')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path",default = 'faces_emore', type = str)
    args = parser.parse_args()

    cfg = configurations[1]
    rec_path = args.rec_path
    rec_path = Path('G:/Dataset/img/faces_webface_112x112')
    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    #load_mx_rec(rec_path)
    
    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    
    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i] + '.bin'), rec_path/bin_files[i], train_transform)