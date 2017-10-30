import argparse
import cv2
import os,errno
import random
import shutil
import util.image
from util.align_dlib import AlignDlib

file_dir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(file_dir, '..', 'model')
dlib_model_dir = os.path.join(modelDir, 'dlib')


def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def alignMain(args):
    mkdirP(args.outputDir)
    imgs = list(image.iterImgs(args.inputDir))
    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), args.dlibFacePredictor))

    if args.landmarks == 'outerEyesAndNose':
        landmarkIndices = align_dlib.OUTER_EYES_AND_NOSE
    elif args.landmarks == 'innerEyesAndBottomLip':
        landmarkIndices = align_dlib.INNER_EYES_AND_BOTTOM_LIP
    else:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + "." + args.ext
        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                    outRgbarr = None
            else:
                outRgbarr = align_dlib.align(args.size, rgb, pad=args.pad, ts=args.ts,
                                    landmarkIndices=landmarkIndices, opencv_det=args.opencv_det,
                                    opencv_model=args.opencv_model, only_crop=args.only_crop)
                if outRgbarr is None and args.verbose:
                    print("  + Unable to align.")

            if args.fallbackLfw and outRgbarr is None:
                nFallbacks += 1
                deepFunneled = "{}/{}.jpg".format(os.path.join(args.fallbackLfw,imgObject.cls),imgObject.name)
                shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args.outputDir,imgObject.cls),imgObject.name))

            if outRgbarr is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                i = 0
                for outRgb in outRgbarr:
                    outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                    out_path = os.path.join(outDir, str(i) + "_" + os.path.basename(imgName))
                    cv2.imwrite(out_path, outBgr)
                    i=i+1

    if args.fallbackLfw:
        print('nFallbacks:', nFallbacks)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--opencv-det', action='store_true', default=False,
                        help='True means using opencv model for face detection(because sometimes dlib'
                             'face detection will failed')
    parser.add_argument('--opencv-model', type=str, default='../model/opencv/cascade.xml',
                        help="Path to dlib's face predictor.")
    parser.add_argument('--only-crop', action='store_true', default=False,
                        help='True : means we only use face detection and crop the face area\n'
                             'False : both face detection and then do face alignment')
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    alignmentParser = subparsers.add_parser(
        'align', help='Align a directory of images.')
    alignmentParser.add_argument('landmarks', type=str,
                                 choices=['outerEyesAndNose', 'innerEyesAndBottomLip'],
                                 help='The landmarks to align to.')
    alignmentParser.add_argument(
        'outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--pad', type=float, nargs='+', help="pad (left,top,right,bottom) for face detection region")
    alignmentParser.add_argument('--ts', type=float, help="translation(,ts) the proportion position of eyes downward so that..."
                                                        " we can reserve more area of forehead",
                                 default='0')
    alignmentParser.add_argument('--size', type=int, help="Default image size.",
                                 default=128)
    alignmentParser.add_argument('--ext', type=str, help="Default image extension.",
                                 default='jpg')
    alignmentParser.add_argument('--fallbackLfw', type=str,
                                 help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")
    alignmentParser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    alignMain(args)