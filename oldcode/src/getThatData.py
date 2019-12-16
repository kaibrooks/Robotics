import os
# import opencv2 as opc
import numpy as np
import csv

def gather_data_and_groundTruth(trainPath, image_type = None):
    if image_type is None:
        image_type = ".jpg"
    
    trainDir = os.listdir(trainPath) # A Directory containing images and a gt csv
    testDir = os.listdir(testPath)

    # Allocate training array
    X_train = np.zeros(len(trainDir))
    X_test = np.zeros(dim=(len(testDir))

    # Open the train and test ground truth csv'
    Y_train = np.genfromtxt(os.path.join(trainPath, "groundTruth.csv"))
    Y_test = np.genfromtxt(os.path.join(testDir, "groundTruth.csv"), delimiter = ",", names = True)
                                                    
    # Gather the training images
    for idx, picture in enumerate(trainDir):
            
            if picture[-4:] == image_type: 
                print("Successfully entered immage preprocessing sub-routine")
                print("Image: " + picture)
                Extract the 3 color channels as well as crop the image_type
                if needed do color/contrast correction. p
                pic_obj = preproc_class(picture)
                pic_obj.crop([dim])
                pic_obj.color_correct(corrections)
                X_train[idx] = pic_obj.output()
                

       # Gather the training images
    for idx, picture in enumerate(testDir):
    
            if picture[-4:] == image_type:
                pic_obj = preproc_class(picture)
                pic_obj.crop([dim])
                pic_obj.color_correct(corrections)
                X_test[idx] = pic_obj.output()
    
    return  X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    gather_data_and_groundTruth(r"../../documentation/images/robot_pix")
