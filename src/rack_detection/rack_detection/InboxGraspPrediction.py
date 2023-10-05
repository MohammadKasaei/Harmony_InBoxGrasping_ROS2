
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry 
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor

import os

class InboxGraspPrediction():
    def __init__(self,sam_model= "vit_b", device = "cuda" ) -> None:

        self._sam_model_type = sam_model
        self._device = device
        if self._sam_model_type == "vit_b":   # 375 MB         
            self._sam_checkpoint = 'models/sam_vit_b_01ec64.pth'
        elif self._sam_model_type == "vit_h": # 2.6 GB           
            self._sam_checkpoint = 'models/sam_vit_h_4b8939.pth'
        else: #1.2 GB
            self._sam_checkpoint = 'models/sam_vit_l_0b3195.pth'

        if not os.path.exists(self._sam_checkpoint):
            print ("model file is not available, check the instruction in the readme.md ...")
            return 

        self._device = device

        self._sam = sam_model_registry[self._sam_model_type](checkpoint=self._sam_checkpoint)
        self._sam.to(device=self._device)

        self._mask_generator1 = SamAutomaticMaskGenerator(self._sam, points_per_batch=16)
        self._predictor = SamPredictor(self._sam)

        self.image = None
        # self.config()

    def find_box_centre(self,vis_masks = False,vis_output=False,color_picker = False):
        # Define the region of interest (ROI) coordinates
        x_offset = 200  # starting x-coordinate
        y_offset = 10  # starting y-coordinate
        width  = 700  # width of the ROI
        height = 350  # height of the ROI
        
        # Crop the image using numpy array slicing
        image = self.image[y_offset:y_offset +height, x_offset :x_offset +width].copy()
        if vis_masks:
            cv2.imshow('image', image)
            # cv2.waitKey(0)

        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(result, (3, 3), 0)

        # Apply Canny edge detection
        # result = cv2.Canny(blurred, 50, 150)   
        result = cv2.Canny(blurred, 50, 200)   
        
        
        if vis_masks:
            cv2.imshow('edge', result)            
       
        # Define the kernel for dilation
        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size according to your needs
        # Perform dilation
        dilated = cv2.dilate(result, kernel, iterations=3)
        if vis_output:
            cv2.imshow('dilated', dilated)


        # Define the structuring element for erosion
        # Here, we're using a 5x5 rectangle
        kernel = np.ones((3,3), np.uint8)

        # Apply erosion
        eroded_image = cv2.erode(dilated, kernel, iterations=13)
        if vis_output:
            cv2.imshow('eroded', eroded_image)
        
            # cv2.waitKey(0)

        #  Find contours in the mask
        contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, contours, -1, (255, 0, 0), 3)  # Drawing all contours in green color


        # # Define a perimeter threshold
        # min_perimeter = 100
        # max_perimeter = 500

        filtered_contours = []
        for contour in contours:
            # perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            # Filtering conditions
            # if min_perimeter < perimeter < max_perimeter and 0.9 < aspect_ratio < 1.1:  # Adjust as needed
            if  0.5 < aspect_ratio < 2:  # Adjust as needed
                filtered_contours.append(contour)

        # # Define a size threshold (in terms of contour area)
        min_contour_area = 3000  # Example value, adjust as needed
        max_contour_area = 80000  # Example value, adjust as needed
        # Filter contours by size
        filtered_contours2 = [contour for contour in filtered_contours if min_contour_area < cv2.contourArea(contour) < max_contour_area]
        

        if filtered_contours2 != []:
            # Find the largest contour by area
            largest_contour = max(filtered_contours2, key=cv2.contourArea)

            # Draw the largest contour on a copy of the original image
            # output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored drawing

            cv2.drawContours(image, [largest_contour], 0, (0, 0, 255), 3)  # Drawing in green color
            cv2.drawContours(image, filtered_contours2, -1, (0, 255, 0), 1)  # Drawing all contours in green color


            # find the center of mask
            # Initialize variables for centroid calculation
            # moments = cv2.moments(contours[0])
            moments = cv2.moments(largest_contour)
            
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            # Draw the center on the mask
            radius = 10
            image = np.ascontiguousarray(image, dtype=np.uint8)


            cv2.circle(image, (center_x, center_y), radius, (0, 255, 255), -1)
            cv2.circle(image, (center_x, center_y), radius-2, (255, 0, 255), -1)
            
            if vis_output:
                cv2.imshow('image', image)

            if vis_masks or vis_output:
                # pass
                cv2.waitKey(1)
                # cv2.destroyAllWindows()
            return [x_offset +center_x, y_offset +center_y]
        
        print ("no contour found...")
        return -1

    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self,coords, labels, ax, marker_size=25):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='green', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='red', linewidth=1.25)   
        
    def show_box(self,box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

    def config_params_based_on_box_centre(self,box_centre):
        input_point1 = np.array([box_centre[0]-25,box_centre[1]-30]).reshape(1,2)
        step_x = 15
        step_y = 5
        for i in range(5):
            for j in range(7):
                input_point1 = np.vstack((input_point1,(input_point1[0,0]+i*step_x,input_point1[0,1]+j*step_y)))
        
        self._input_point = input_point1        
        self._input_label = np.ones(36)
        

    def generate_masks(self,dbg_vis = False):
        # self.image = cv2.imread(image_path)
        
        center=self.find_box_centre(vis_masks=dbg_vis,vis_output=dbg_vis)
        if center !=-1:
            self.config_params_based_on_box_centre(center)
        else:
            print ("can not find the centre of the box...")
            return [],[]


        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        self._predictor.set_image(self.image)
        self._masks, self._scores, self._logits = self._predictor.predict(
            point_coords=self._input_point,
            point_labels=self._input_label,
            # multimask_output=True,
            # box=input_box[None, :],
            multimask_output=True,
        )

        mask_input = self._logits[np.argmax(self._scores), :, :]  # Choose the model's best mask

        self._masks, _, _ = self._predictor.predict(
            point_coords=self._input_point,
            point_labels=self._input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

        return self._masks, self._scores
    

    def generate_grasp(self,mask,vis=True):

        (contours, hierarchy) = cv2.findContours(np.uint8(mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 3 
        j = 0
        grasp_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500 and area < 20000:
                rect = cv2.minAreaRect(contour)
                _, _, angle = rect
                angle = 90-angle if angle>45 else -angle 
                

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center = np.int16((np.mean(box[:,0]),np.mean(box[:,1])))

                tmp = box.tolist() 
                bs =np.array(sorted(tmp, key=lambda a_entry: a_entry[0]))
               
                center1 = np.int16(((bs[0,0]+bs[1,0])/2,(bs[0,1]+bs[1,1])/2))        
                center2 = np.int16(((bs[2,0]+bs[3,0])/2,(bs[2,1]+bs[3,1])/2))        
                
                grasp_list.append ([center,center1,center2,angle])
                if vis:
                    cv2.drawContours(self.image_rgb, contours, j, (255, 255, 0), thickness)                
                    cv2.drawContours(self.image_rgb,[box],0,(0,255,255),thickness)
                    cv2.circle(self.image_rgb, center=center, radius=10, color = (255,255,255), thickness=-1) 
                    cv2.circle(self.image_rgb, center=center1, radius=10, color = (255,0,255), thickness=5) 
                    cv2.circle(self.image_rgb, center=center2, radius=10, color = (255,0,255), thickness=5) 
            j += 1

        if grasp_list == [] :
            for contour in contours:
                area = cv2.contourArea(contour) 
                print (f"contour area:{area}")
        

        return grasp_list

