from utils.evaluate import evaluate_cm
from utils.load_model import Model

test_set_path = "/home/thien/Desktop/crack_detection/ImageClassification/dataset/test"
model = Model("/home/thien/Desktop/crack_detection/models_resnet/ResNet50")
evaluate_cm(model, test_set_path, normalize=False)