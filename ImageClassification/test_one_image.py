from utils.load_model import Model
#from utils.plot import plot_image
import cv2

model = Model("/home/thien/Desktop/crack_detection/models_resnet/ResNet34")
image = cv2.imread("/home/thien/Desktop/crack_detection/ImageClassification/balance_dataset/Data/error_train/cracked/130.jpg")
pred = model.predict(image)
print(pred)