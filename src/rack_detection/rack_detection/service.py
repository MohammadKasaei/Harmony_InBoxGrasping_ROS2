import rclpy
from rclpy.node import Node

from rack_detection_interface.srv import RackDetection

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from segment_anything import sam_model_registry 
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
import datetime


from sensor_msgs.msg import CameraInfo

cv_bridge = CvBridge()

# in my laptop we need to set this param to get it run on the GPU, not sure if we need this always.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" 

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')

        self.rgb_image   = None
        self.depth_image = None

        rgb_subscription = self.create_subscription(
        CompressedImage,
        "/myumi_005/sensors/top_azure/rgb/image_raw/compressed",  # Replace with your RGB compressed topic
        self.rgb_image_callback,
        1,
        )

        depth_subscription = self.create_subscription(
            Image,
            # "/myumi_005/sensors/top_azure/depth/image_raw",  # Replace with your depth topic
            "/myumi_005/sensors/top_azure/depth_to_rgb/image_raw",
            self.depth_image_callback,
            1,
        )

        camera_info_subscription = self.create_subscription(CameraInfo, '/myumi_005/sensors/top_azure/rgb/camera_info', self.camera_info_callback, 10)

        torch.cuda.empty_cache()
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        self.vision = InboxGraspPrediction()
        self.srv = self.create_service(RackDetection, 'rack_detection', self.rack_detection_callback)


    def camera_info_callback(self, msg: CameraInfo):
        # print ("camera:", msg.k)
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        # Use these parameters as needed



    def rack_detection_callback(self, request, response):
        
        rack_pos = Pose()
        if (self.rgb_image is None) or (self.depth_image is None):
            rack_pos.position.x = -1.0
            rack_pos.position.y = -1.0
            rack_pos.position.z = -1.0
            response.probablity = 0.0
            response.pose       = rack_pos        
            response.execute    = 0

            return response
        
        raw_image = self.rgb_image.copy()      
        # raw_depth_image = self.depth_array.copy()      
        raw_depth_image = self.depth_image.copy()      
        
        self.vision.image = self.rgb_image.copy()      
        self.grasp_center = None

        masks, scores = self.vision.generate_masks(dbg_vis=request.dbg_vis)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.subplot(121)
            plt.imshow(self.vision.image_rgb)
            plt.title('Original')
            plt.axis('on')
            plt.subplot(122)
            # plt.imshow(gs.image)
            plt.title('Grasp')
            if request.show_masks_point:
                self.vision.show_mask(mask, plt.gca(),random_color=False)
                self.vision.show_points(self.vision._input_point, self.vision._input_label, plt.gca())
            gs_list = self.vision.generate_grasp(mask,vis=True)
            if len(gs_list)>0:
                self.grasp_center = gs_list[0][0]

            print ("grasp list:\n", gs_list)
            plt.imshow(self.vision.image_rgb)
            plt.axis('on')
            plt.show()

        print ("done")
        key = 0

        if request.show_raw_image:
            cv2.imshow("RGB Image", raw_image)
            # cv2.imshow("Depth Image", self.depth_image)
            key = cv2.waitKey(0)
            if key == ord('s') or key ==ord('S'):
                if not os.path.isdir('capture_images'):
                    os.makedirs('capture_images')
                    print ("directory is creasted.")
                current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"capture_images/{current_time}_captured.jpg"
    
                cv2.imwrite(filename, raw_image)
                print ("image saved.")
           
        cv2.destroyAllWindows()
        
        
        

        print ("depth size:",self.depth_image.shape)
        print ("image size:",self.rgb_image.shape)
        # print ("depth array:",self.depth_array.shape)
        
        # Extract depth value for the target pixel position
        if self.grasp_center is not None:
            depth_pixel_x = int(self.grasp_center[0])
            depth_pixel_y = int(self.grasp_center[1])

            

            # distance_arr = raw_depth_image[depth_pixel_y-5:depth_pixel_y+5,depth_pixel_x-5:depth_pixel_x+5][0]
            # distance_arr = distance_arr.flatten()
            # median = np.median(distance_arr)

            # print (f"median: {median:3.3f}")



            # depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x][0])
            depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x])
            
            depth_in_meters = depth_value / 1000.0
            print (f"depth [{depth_pixel_x},{depth_pixel_y}] = {depth_value}")
            #  float(frame[y_center][x_center])*100.0

            
            rack_pos.position.x = float(self.grasp_center[0])
            rack_pos.position.y = float(self.grasp_center[1])
            rack_pos.position.z = depth_in_meters
            response.probablity  = 1.0

            self.pixel_to_meter (self.grasp_center[0],self.grasp_center[1],depth_in_meters)

        else:
            rack_pos.position.x = -1.0
            rack_pos.position.y = -1.0
            rack_pos.position.z = -1.0
            response.probablity  = 0.0
             

        response.pose = rack_pos        
        response.execute = 1 if (key== ord('e') or key ==ord('E')) else 0

        if response.execute :
            print (self.depth_array)

        # response.sum = request.dbg_vis + request.show_masks_point + request.show_raw_image
        # self.get_logger().info('Incoming request\na: %d b: %d  c: %d' % (request.dbg_vis, request.show_masks_point,request.show_raw_image))

        torch.cuda.empty_cache()

        return response
    

    def pixel_to_meter(self,x_pixel, y_pixel, depth):
        """
        Convert pixel coordinates and depth to a 3D point in camera's coordinate system (in meters).

        :param x_pixel: x coordinate of the pixel
        :param y_pixel: y coordinate of the pixel
        :param depth: depth value at that pixel (in meters)
        :param fx: focal length in x direction (in pixels)
        :param fy: focal length in y direction (in pixels)
        :param cx: principal point x-coordinate (in pixels)
        :param cy: principal point y-coordinate (in pixels)
        :return: (X_cam, Y_cam, Z_cam) coordinates in camera's frame (in meters)
        """
        x_cam = (x_pixel - self.cx) * depth / self.fx
        y_cam = (y_pixel - self.cy) * depth / self.fy
        z_cam = depth

        print ("camera", x_cam,y_cam,z_cam)
        
        return x_cam, y_cam, z_cam


    def rgb_image_callback(self,msg):
        try:
            self.rgb_image = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"Error: {str(e)}")

    def depth_image_callback(self,msg):

        try:
            self.depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")          
        except Exception as e:
            print(f"Error: {str(e)}")


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
        
    def config(self):
        
        input_point1 = np.array([330,210]).reshape(1,2)
        step_x = 10
        step_y = 12
        for i in range(3):
            for j in range(5):
                input_point1 = np.vstack((input_point1,(input_point1[0,0]+i*step_x,input_point1[0,1]+j*step_y)))
        
        self._input_point = input_point1        
        self._input_label = np.ones(22)
        

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
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center = np.int16((np.mean(box[:,0]),np.mean(box[:,1])))

                tmp = box.tolist() 
                bs =np.array(sorted(tmp, key=lambda a_entry: a_entry[0]))
               
                center1 = np.int16(((bs[0,0]+bs[1,0])/2,(bs[0,1]+bs[1,1])/2))        
                center2 = np.int16(((bs[2,0]+bs[3,0])/2,(bs[2,1]+bs[3,1])/2))        
                
                grasp_list.append ([center,center1,center2])
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


def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()