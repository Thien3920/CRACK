from utils.load_model import Model
import cv2
import numpy as np
from glob import glob
import shutil
import os

def export_error_one_class(model,path_image,threshold=50,object = 'cracked'):
    classes = {
                "cracked":0,
                "uncracked":1,
              }
    list_images = glob(path_image + '/*.jpg') + glob(path_image + '/*.jpeg') + glob(path_image + '/*.png')
    list_images.sort()
    total_imges = len(list_images)

    A = np.zeros(shape=[total_imges,3])
    A = np.array(A,dtype = str)

    for i,image_path in enumerate(list_images):
        image = cv2.imread(image_path)
        pred = model.predict(image)
        image_name = image_path.split('/')[-1]
        score = pred[1][classes[object]]
        A[int(i),:] = [object,score,image_name]

    # get another images with  score <= threshold
    error_image = A[ A[:,1]<=str(threshold) ]
    return error_image

def export_error_test(model,input_path,output_path,threshold = 50):
    crack_path = os.path.join(input_path,'cracked')
    uncrack_path = os.path.join(input_path,'uncracked')

    crack_error =  export_error_one_class(model,crack_path,threshold,object = 'cracked')
    uncrack_error = export_error_one_class(model, uncrack_path, threshold, object = 'uncracked')

    if not  os.path.exists(output_path):
        for i in ['','cracked','uncracked']:
            os.mkdir(os.path.join(output_path,i))
    for folder in [crack_error,uncrack_error]:
        for i in range(folder.shape[0]):
            source = os.path.join(input_path,folder[i,0],folder[i,2])
            destination = os.path.join(output_path,folder[i,0],folder[i,2])
            try:
                shutil.copy(source, destination)
            except:
                print("Incorrect source or destination")



if __name__ == "__main__":

    model = Model("/home/thien/Desktop/crack_detection/models_resnet/Resnet101")
    input_path = '/home/thien/Desktop/crack_detection/ImageClassification/dataset/train'
    output_path = './Data/error_train'
    export_error_test(model,input_path,output_path,threshold=30)