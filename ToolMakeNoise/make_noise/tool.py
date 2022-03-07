import numpy as np
import cv2
import random
import imutils
import os
import glob


def brightness(img, alpha):
    """
        change brightness image
        alpha = 0 : 250
    """
    img_new = np.asarray(img + alpha)   # cast pixel values to int
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new


def rotation(image, angle):
    """
        rotation image
        cv2 : can only rotate 90, 180, 270, ...
        imutils: can rotate all angles (image size will be changed)
    """
    angle_list = [cv2.cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    if (angle % 360) == 0:
        rotated = image
    elif (angle % 90) == 0:
        index = int((angle / 90) % 4) - 1
        rotated = cv2.rotate(image, angle_list[index])
    else:
        rotated = imutils.rotate_bound(image, angle)

    return rotated


# vertical or horizontal
def flip(image, code):
    """
    code:
        0: vertical
        1 (+): horizontal
        -1 (-): both
    """
    result = cv2.flip(image, code)
    return result


def blur(image, level):
    level = np.array(level)
    if level.shape == ():
        if level == 0:
            x_blur = 1
            y_blur = 1
        else:
            x_blur = 2*level
            y_blur = 2*level
    elif level.shape == (2,):
        [x_blur, y_blur] = level
    else:
        print("Error in blur image because level no invalid !!")
        return image
    result = cv2.blur(image, (x_blur, y_blur))
    return result


def even(num):
    if num % 2 == 0:
        return num
    else:
        return num - 1


def random_position(img, mark, t):
    x_img = even(img.shape[0])
    y_img = even(img.shape[1])
    x_mark = even(mark.shape[0])
    y_mark = even(mark.shape[1])

    if (x_img == 0) | (y_img == 0):
        return img
    if (x_mark == 0) | (y_mark == 0):
        return img

    x = random.randint(2, int(x_img/2))*2-2
    y = random.randint(2, int(y_img/2))*2-2

    xms = 0
    xme = x_mark
    yms = 0
    yme = y_mark

    xis = x - int((xme - xms)/2)
    xie = x + int((xme - xms)/2)
    yis = y - int((yme - yms)/2)
    yie = y + int((yme - yms)/2)

    if x < x_mark/2:
        xms = int(x_mark/2 - x)
        xis = 0
    if (x_img - x) < (x_mark/2):
        xme = int(x_mark/2 + (x_img - x))
        xie = x_img
    if y < y_mark / 2:
        yms = int(y_mark / 2 - y)
        yis = 0
    if (y_img - y) < (y_mark / 2):
        yme = int(y_mark / 2 + (y_img - y))
        yie = y_img

    if t == 'add':
        img[xis:xie, yis:yie] = cv2.add(img[xis:xie, yis:yie], mark[xms:xme, yms:yme])
    if t == 'subtract':
        img[xis:xie, yis:yie] = cv2.subtract(img[xis:xie, yis:yie], mark[xms:xme, yms:yme])
    return img


def noise(image, path_img_noise, ness, math, scale=1):
    # random type rain
    list_rain_type = glob.glob(path_img_noise + '/*.png')
    rain_type_img = random.choice(list_rain_type)
    rain_drop_img = cv2.imread(rain_type_img)
    # change brightness
    rain_drop_img = brightness(rain_drop_img, ness)
    # scale image
    [x, y, z] = rain_drop_img.shape
    [x_new, y_new] = [int(x*scale), int(y*scale)]
    rain_drop_img = cv2.resize(rain_drop_img, [y_new, x_new])
    cv2.imwrite('temp.jpg', rain_drop_img)  #
    rain_drop_img = cv2.imread('temp.jpg')  # need to be improved
    os.remove('temp.jpg')                   #
    image = random_position(image, rain_drop_img, math)
    return image


def raindrop(image, level=30, scale_level=1.5):
    for i in range(level):
        rain_drop_path = 'I:/More/AI/Projects/CrackDetection/ImageClassification-master-update/noise/raindrop'
        ness = random.randint(-30, -20)
        image = noise(image, rain_drop_path, ness, 'add', scale_level)
    return image


def smoke(image, scale_level=3):
    path_smoke_noise = 'I:/More/AI/Projects/CrackDetection/ImageClassification-master-update/noise/smoke'
    # change brightness
    ness = random.randint(-30, -10)
    return noise(image, path_smoke_noise, ness, random.choice(['add', 'subtract']), scale_level)


def dust(image, scale_level=3):
    path_smoke_noise = 'I:/More/AI/Projects/CrackDetection/ImageClassification-master-update/noise/dust'
    # change brightness
    ness = random.randint(-50, -40)
    return noise(image, path_smoke_noise, ness, 'add', scale_level)


def rain(image, scale_level=2):
    # random type rain
    path_rain_noise = 'I:/More/AI/Projects/CrackDetection/ImageClassification-master-update/noise/rain'
    # change brightness
    ness = random.randint(-220, -200)
    return noise(image, path_rain_noise, ness, 'subtract', scale_level)


# def sun(image):
#     # random type sun
#     # path_sun_noise = 'I:/More/AI/Projects/CrackDetection/ImageClassification-master-update/noise/sun'
#     # change brightness
#     # ness = random.randint(-200, -180)
#     # image = cv2.imread('./new.jpg')
#     B = image[:, :, 0]
#     G = image[:, :, 1]
#     R = image[:, :, 2]
#
#     rnd = random.randint(10, 40)
#     red_rnd = random.randint(5, rnd)
#     image[:, :, 0] = B - 3*rnd
#     image[:, :, 1] = G - red_rnd
#     image[:, :, 2] = R
#     return image


def sun(image):
    # random type sun
    path_sun_noise = 'I:/More/AI/Projects/CrackDetection/ImageClassification-master-update/noise/sun'
    list_sun_type = glob.glob(path_sun_noise + '/*.png')
    sun_type_img = random.choice(list_sun_type)
    sun_img = cv2.imread(sun_type_img)
    # change brightness
    ness = random.randint(-210, -185)
    sun_img = brightness(sun_img, ness)
    cv2.imwrite('temp.jpg', sun_img)  #
    sun_img = cv2.imread('temp.jpg')  # need to be improved
    os.remove('temp.jpg')
    image[0:255, 0:255] = cv2.add(image[0:255, 0:255], sun_img[0:255, 0:255])
    return image


def salt_pepper_noise(image, prob=5):
    black = prob/200
    output = np.zeros(image.shape, np.uint8)
    white = 1 - black
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < black:
                output[i][j] = 0
            elif rdn > white:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
