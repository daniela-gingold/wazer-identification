from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from matplotlib import pyplot
from mrcnn.model import mold_image
from numpy import expand_dims
from matplotlib.patches import Rectangle
from keras.models import load_model
from urllib.request import Request as URLRquest
from remove_logo import remove_logo
from urllib.request import urlopen
import timeit, argparse
import numpy as np
import cv2
import os

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "wazer_cfg"
	# number of classes (background + wazer + logo)
	NUM_CLASSES = 1 + 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# returns None if rectangles don't intersect, otherwise returns intersection area
def area_overlap(a_ymin, a_xmin, a_ymax ,a_xmax, b_ymin, b_xmin, b_ymax ,b_xmax):
    dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy

def area(ymin, xmin, ymax , xmax):
    dx = xmax - xmin
    dy = ymax - ymin
    return dx * dy

def get_wazers(image, model, cfg):

    # convert pixel values
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]

    boxes = yhat['rois']
    class_ids = yhat['class_ids']
    scores = yhat['scores']

    num_of_predictions = len(class_ids)

    # init class lists
    list_wazers = []
    list_logos = []

    for prediction_index in range(num_of_predictions):
        box = boxes[prediction_index]
        prediction = class_ids[prediction_index]
        score = round(scores[prediction_index] * 100, 2)

        #get coordinates
        y1, x1, y2, x2 = box

        if prediction == 1:
            #wazer
            list_wazers.append([y1, x1, y2, x2, score])
        elif prediction == 2:
            # logo
            list_logos.append([y1, x1, y2, x2, score])

    # exclude wazers which are part of a logo
    if len(list_logos) > 0:
        logo = list_logos[0]
        logo_y1, logo_x1, logo_y2, logo_x2, _ = logo

        # init index of wazer inside logo
        wazer_in_logo_index = -1

        # cropped_logo = image_crop(image, logo_y1, logo_x1, logo_y2, logo_x2)
        # cv2.imshow('cropped_logo', cropped_logo)
        # cv2.waitKey(0)


        # iterate over each wazer and check if it overlaps selected logo
        for i in range(len(list_wazers)):
            wazer = list_wazers[i]
            wazer_y1, wazer_x1, wazer_y2, wazer_x2, _ = wazer

            # cropped_wazer = image_crop(image, wazer_y1, wazer_x1, wazer_y2, wazer_x2)
            # cv2.imshow('cropped_wazer', cropped_wazer)
            # cv2.waitKey(0)

            overlap_area = area_overlap(logo_y1, logo_x1, logo_y2, logo_x2, wazer_y1, wazer_x1, wazer_y2, wazer_x2)
            if overlap_area != None:
                wazer_area = area(wazer_y1, wazer_x1, wazer_y2, wazer_x2)

                cover_ratio = ((overlap_area * 100) / wazer_area)

                if cover_ratio > 50:
                    wazer_in_logo_index = i
                    break

        if wazer_in_logo_index != -1:
            list_wazers.pop(wazer_in_logo_index)
            print('     wazer in logo identified...')

    return list_wazers

def image_preprocessing(image):
    data = []

    dim = (32, 32)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    data.append(resized)

    arr = np.array(data)

    # normalize
    arr = arr.astype('float32') / 255

    # subtract pixel mean
    image_mean = np.mean(arr)
    arr -= image_mean

    return arr

def predict_by_image(image_path, model_1, cfg, model_2, use_match_template):

    if image_path.startswith('http'):
        # in case of url path
        image = url_to_image(image_path)
    else:
        # in case of local path
        image = cv2.imread(image_path)

    # verify image load
    if image is None:
        return None

    if use_match_template:
        #try to remove logo from image
        image = remove_logo('logo_1.png', image)

    list_wazers = get_wazers(image, model_1, cfg)

    wazers_size = len(list_wazers)

    prediction = False

    if wazers_size >= 0:

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # print(image.shape)
        # cv2.imshow('image_after_COLOR_BGR2RGB', image)
        # cv2.waitKey(0)

        for wazer in list_wazers:

            # get coordinates and score
            y1, x1, y2, x2, score = wazer
            cropped_image = image[y1:y2, x1:x2]
            # cv2.imshow('cropped_image', cropped_image)
            # cv2.waitKey(0)
            # cv2.imwrite('crop.jpg', cropped_image)
            input_image = image_preprocessing(cropped_image)
            prediction_score = model_2.predict(input_image)
            prediction_threshold = 0.5

            if prediction_score > prediction_threshold:
                prediction = True
                break

    return prediction

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
    try:
        req = URLRquest(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urlopen(req)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        return image
    except:
        print('!!!      Error while trying to download image from {}        !!!'.format(url))
        return None

def run(test_image_path, model_path, secondary_model_path, use_match_template = False):

    # create config
    cfg = PredictionConfig()

    # define first model
    model_1 = MaskRCNN(mode='inference', model_dir='./', config=cfg)

    # load model weights
    model_1.load_weights(model_path, by_name=True)

    #define secondary model
    model_2 = load_model(secondary_model_path)

    result = predict_by_image(test_image_path, model_1, cfg, model_2, use_match_template)

    if result:
        return 'wazer'
    elif result==None:
        return 'image was not found...'
    else:
        return 'no wazer'

def str_to_bool(val):
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_path", type=str, default='image.jpg', help="test image path")
    parser.add_argument("--model_path", type=str, default='model_1.h5', help="first model path")
    parser.add_argument("--secondary_model_path", type=str, default='models_2.h5', help="secondary model path")
    parser.add_argument("--use_match_template", type=str, default=False, help="use match template flag")
    opt = parser.parse_args()

    opt.use_match_template = str_to_bool(opt.use_match_template)

    print('processing {}...'.format( opt.test_image_path))

    # run test
    start = timeit.default_timer()
    result = run(opt.test_image_path, opt.model_path, opt.secondary_model_path, opt.use_match_template)
    stop = timeit.default_timer()

    # print results
    print('{} - {}'.format(opt.test_image_path, result))
    print('Run Time: ', stop - start)
