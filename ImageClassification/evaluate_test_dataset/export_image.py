from utils.load_model import Model
from glob import glob
import numpy as np
import argparse
import cv2
import os

def export_max_min_medium_score(model,path_image,n_images=2,object = 'cracked',img_size =[600,600]):
    classes = {
                "cracked":0,
                "uncracked":1
              }
    list_images = glob(path_image + '/*.jpg') + glob(path_image + '/*.jpeg') + glob(path_image + '/*.png')
    list_images.sort()
    total_imges = len(list_images)
    print("Number image: ", total_imges)

    A = np.zeros(shape=[total_imges,3])
    A =  np.array(A,dtype = str)

    for i,image_path in enumerate(list_images):
        image = cv2.imread(image_path)
        pred = model.predict(image)
        image_name = image_path.split('/')[-1]
        score = pred[1][classes[object]]
        score = score - 0.00001 # avoid the case score = 100%
        A[int(i),:] = [object,score,image_name]
    A = A[A[:, 1].argsort()]


    max_scores = A[-n_images:A.shape[0], :]
    min_scores = A[0:n_images, :]

    # get another images with  40% <= score <= 60%
    medium_scores = A[ A[:,1] >=str(40) ]
    medium_scores = medium_scores[medium_scores[:, 1] <= str(60)]

    if medium_scores.shape[0] > n_images:
        medium_scores = medium_scores[0:n_images,:]

    return max_scores, min_scores, medium_scores

def concat_images(cracked_path,max_cracked,min_cracked,medium_cracked,uncracked_path,max_uncracked,min_uncracked,medium_uncracked):
    max_crack = h_concat_images(cracked_path, max_cracked, object = "cracked" )
    min_crack = h_concat_images(cracked_path, min_cracked, object = "cracked")
    medium_crack = h_concat_images(cracked_path, medium_cracked, object = "cracked")

    max_uncrack = h_concat_images(uncracked_path, max_uncracked, object="uncracked")
    min_uncrack = h_concat_images(uncracked_path, min_uncracked, object="uncracked")
    medium_uncrack = h_concat_images(uncracked_path, medium_uncracked,object="uncracked")

    cols = max_crack.shape[1]
    if (medium_crack.shape[1] == cols and medium_uncrack.shape[1] == cols):
        v_concat = cv2.vconcat([max_crack,min_crack,medium_crack,max_uncrack,min_uncrack,medium_uncrack])
        cv2.imwrite('Img.jpg', v_concat)
    else:
        v_concat = cv2.vconcat([max_crack, min_crack, max_uncrack, min_uncrack])
        cv2.imwrite('Img.jpg', v_concat)

        if (medium_crack.shape[1] == medium_uncrack.shape[1]):
            medium_img = cv2.vconcat([medium_crack, medium_uncrack])
            cv2.imwrite('medium_Img.jpg', medium_img)


def h_concat_images(folder_path,A,object,img_size=(600,600)):
    concat = []
    for i in range(A.shape[0]):
        img_path = os.path.join(folder_path,A[i,2])
        Image = cv2.imread(img_path)
        Image = cv2.resize(Image, dsize=img_size)
        score = A[i,1][0:5] # Round to Two Decimal Places

        cv2.putText(Image,object + ': ' + score , (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        concat.append(Image)
    Img = cv2.hconcat(concat)
    return Img

def export_image(model,path):
    cracked_path = os.path.join(path, "cracked")
    uncracked_path = os.path.join(path, "uncracked")

    max_cracked, min_cracked, medium_cracked = export_max_min_medium_score(model, cracked_path, n_images=5,object='cracked')
    max_uncracked, min_uncracked, medium_uncracked = export_max_min_medium_score(model, uncracked_path, n_images=5,object='uncracked')

    concat_images(cracked_path, max_cracked, min_cracked, medium_cracked, uncracked_path, max_uncracked,
                  min_uncracked, medium_uncracked)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/thien/Desktop/crack_detection/models_resnet/Resnet152', help='model path')
    parser.add_argument('--test_dir_path', default='/home/thien/Desktop/crack_detection/ImageClassification/dataset/test', help='test directory path')
    args = parser.parse_args()

    # Load model
    model = Model(args.model)

    path = args.test_dir_path
    export_image(model, path)
    print('Done!')