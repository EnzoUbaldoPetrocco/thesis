#! /usr/bin/env python3
import cv

class Detectron:

    
    def crop_image_from_mask(self, mask, image):
            '''
            Filter the original image with the mask of the PCB, so to have the original full size resolution image with only
            the PCB visible over a black background

            Args: 
                mask(Mat): the mask of the PCB
                image(Mat): the original image in OpenCV format

            '''

            masked = cv.bitwise_and(image,image,mask = mask)
            print("Shape: "+str(masked.shape))
            masked_tr=cv.cvtColor(masked, cv.COLOR_BGRA2BGR)
            cv.imwrite(join(self.log_dir, "mask"+str(self.count_mask)+".jpg"), masked_tr)
            self.count_mask+=1
            
            return masked_tr

        