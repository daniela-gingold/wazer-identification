import numpy as np
import imutils
import cv2

def log(output, debug):
    if debug:
        print(output)

def remove_logo(template_path, image, debug=False):

    # load the template, convert it to grayscale, and detect edges
    template = cv2.imread(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 40, 120) # below minVal are sure to be non-edges, above maxVal are sure to be edges
    (tH, tW) = template.shape[:2]
    # cv2.imshow("template", template)

    (x, y, z) = image.shape 
    log(image.shape, debug)

    # in case of a large image
    RESIZE_THRESHOLD = 1000
    if x > RESIZE_THRESHOLD or y > RESIZE_THRESHOLD:
        x = int(x / 2)
        y = int(y / 2)
        # resize image
        dim = (y, x)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        log('resize', debug)
        log(image.shape, debug)

    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.00625, 2.5, 200)[::-1]:

        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template,
        # then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 40, 120)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)

        # minimum correlation value,  maximum correlation value,
        # (x, y)-coordinate of the minimum value, (x, y)-coordinate of the maximum value
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute actual (x, y) coordinates
    # of the bounding box based on the resized ratio to ensure that the
    # coordinates match the original dimensions of the input image
    (maxVal, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # treshold for identifying the template
    if maxVal > 0.24:
        if debug:
            # show image and draw rectangle
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, str(round(maxVal, 2)), (startX, startY+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            print('startY:{}, endY:{}, startX:{}, endX:{}, maxVal"{}'.format(startY,endY,startX,endX, maxVal))
        else:
            # replace logo pixels with the same pixel
            image[startY:endY,startX:endX] = 1
            print('     match template identified a logo...')
            # print(image.shape)
            # cv2.imshow('image_after_match_logo', image)
            # cv2.waitKey(0)
    return image

