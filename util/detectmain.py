import util.detection as dtn
import os
from util.symbol.config import config
import argparse


def main() :
    arr = os.listdir('data')
    dirname = os.path.join(os.getcwd(), "data")
    for fl in arr:
        imgpath = os.path.join(dirname, fl)
        dtn.detect(imgpath, args)
    print("detection done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use pre-trainned resnet model to classify one image")
    parser.add_argument('--img', type=str, default='test.jpg', help='input image for classification')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--prefix', type=str, default='mxnet-face-fr50', help='the prefix of the pre-trained model')
    parser.add_argument('--epoch', type=int, default=0, help='the epoch of the pre-trained model')
    parser.add_argument('--thresh', type=float, default=0.8, help='the threshold of face score, set bigger will get more'
                                                                  'likely face result')
    parser.add_argument('--nms-thresh', type=float, default=0.3, help='the threshold of nms')
    parser.add_argument('--min-size', type=int, default=24, help='the min size of object')
    parser.add_argument('--scale', type=int, default=600, help='the scale of shorter edge will be resize to')
    parser.add_argument('--max-scale', type=int, default=1000, help='the maximize scale after resize')

    args = parser.parse_args()
    config.END2END = 1
    config.TEST.HAS_RPN = True
    config.TEST.RPN_MIN_SIZE = args.min_size
    config.SCALES = (args.scale, )
    config.MAX_SIZE = args.max_scale
    main()

