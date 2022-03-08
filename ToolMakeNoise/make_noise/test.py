from make_noise import tool
import cv2


# img = cv2.imread('../new/lisa.png')
# cv2.imshow("root", img)
#
# i = [200, 1]
#
# im = tool.blur(img, i)
# cv2.imshow(str(i), im)

im = cv2.imread('/home/thien/Desktop/crack_detection/ImageClassification/dataset/test/cracked/28.jpg')
# cv2.imshow('a', im)
for i in range(10):
    # if i % 1000 == 0:
    #     print(i)
    # img = tool.noise_raindrop(im.copy(), scale_level=i+1)
    # img = tool.raindrop(im.copy())
    # cv2.imshow('a'+str(i), img)
    img = tool.rain(im.copy(), 4)
    cv2.imshow('a' + str(i), img)
    # img = tool.sun(im.copy())
    # cv2.imshow('a' + str(i), img)
    # img = tool.smoke(im.copy())
    # cv2.imshow('a' + str(i), img)
    # img = tool.dust(im.copy())
    # cv2.imshow('a' + str(i), img)
    # img = tool.salt_pepper_noise(im.copy())
    # cv2.imshow('a' + str(i), img)
cv2.waitKey()
