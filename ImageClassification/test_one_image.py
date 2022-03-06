from utils.load_model import Model
#from utils.plot import plot_image
import cv2

model = Model("/home/thien/Desktop/crack_detection/models_resnet/result_crack_resnet101_ban_dongbo")
image = cv2.imread("/home/thien/Desktop/crack_detection/ImageClassification/dataset/test/cracked/28.jpg")
pred = model.predict(image)
print(pred)