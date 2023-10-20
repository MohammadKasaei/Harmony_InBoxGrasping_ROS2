import rclpy
from rclpy.node import Node

from rack_detection_interface.srv import RackDetection
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo

import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
# import open3d as o3d

import os
import datetime

import torch

from .InboxGraspPrediction import InboxGraspPrediction
# in my laptop we need to set this param to get it run on the GPU, not sure if we need this always.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" 

cv_bridge = CvBridge()

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')

        self.rgb_image   = None
        self.depth_image = None

        rgb_subscription = self.create_subscription(
        CompressedImage,
        "/myumi_005/sensors/top_azure/rgb/image_raw/compressed",  
        self.rgb_image_callback,
        1,
        )

        depth_subscription = self.create_subscription(
            Image,
            "/myumi_005/sensors/top_azure/depth_to_rgb/image_raw",
            self.depth_image_callback,
            1,
        )

        camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/myumi_005/sensors/top_azure/rgb/camera_info',
            self.camera_info_callback,
            10
        )

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

        self.vision.fx = msg.k[0]
        self.vision.fy = msg.k[4]
        self.vision.cx = msg.k[2]
        self.vision.cy = msg.k[5]
        # Use these parameters as needed

    def rack_detection_callback(self, request, response):
        print ("-"*40)
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
        self.vision.depth = self.depth_image.copy()
              
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
                center1 = gs_list[0][1]        
                center2 = gs_list[0][2]        
                center3 = gs_list[0][3]        
                center4 = gs_list[0][4]        
                self.grasp_angle = gs_list[0][5]
                self.box = gs_list[0][6]
                

            print ("grasp list:\n", gs_list)
            plt.imshow(self.vision.image_rgb)
            plt.axis('on')
            
            cv2.waitKey(0)
            plt.show()
            

        print ("done")
        key = 0

        if request.show_raw_image:
            cv2.imshow("RGB Image", raw_image)
            key = cv2.waitKey(0)
            if key == ord('s') or key ==ord('S'):
                if not os.path.isdir('capture_images'):
                    os.makedirs('capture_images')
                    print ("directory is created.")
                current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"capture_images/{current_time}_captured.jpg"
    
                cv2.imwrite(filename, raw_image)
                print ("image saved.")
           
        cv2.destroyAllWindows()

        # print ("depth size:",self.depth_image.shape)
        # print ("image size:",self.rgb_image.shape)
        # print ("box:", self.box)

        # Extract depth value for the target pixel position
        if self.grasp_center is not None:
            depth_pixel_x = int(self.grasp_center[0])
            depth_pixel_y = int(self.grasp_center[1])
            depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x])
            depth_in_meters = depth_value / 1000
            x_cam, y_cam, z_cam = self.pixel_to_meter(self.grasp_center[0],self.grasp_center[1],depth_in_meters)
            idx =0 
            while depth_value<0.01 and idx<10:
                depth_pixel_x = int(self.grasp_center[0])
                depth_pixel_y = int(self.grasp_center[1])
                depth_value = float(raw_depth_image[depth_pixel_y+idx,depth_pixel_x+idx])        
                depth_in_meters = depth_value / 1000
                x_cam, y_cam, z_cam = self.pixel_to_meter(self.grasp_center[0],self.grasp_center[1],depth_in_meters)
            print ("pos:",x_cam, y_cam, z_cam)

            centere3_depth_pixel_x = int(center3[0])
            centere3_depth_pixel_y = int(center3[1])
            depth_value = float(raw_depth_image[centere3_depth_pixel_y,centere3_depth_pixel_x])
            depth_in_meters = depth_value / 1000
            centere3_x_cam, centere3_y_cam, centere3_z_cam = self.pixel_to_meter(center3[0],center3[1],depth_in_meters)

            centere1_depth_pixel_x = int(center1[0]+10)
            centere1_depth_pixel_y = int(center1[1])
            depth_value = float(raw_depth_image[centere1_depth_pixel_y,centere1_depth_pixel_x])
            depth_in_meters = depth_value / 1000
            centere1_x_cam, centere1_y_cam, centere1_z_cam = self.pixel_to_meter(center1[0]+10,center1[1],depth_in_meters)

            length = np.linalg.norm ([centere3_x_cam-x_cam,centere3_y_cam-y_cam]) * 2
            width = np.linalg.norm ([centere1_x_cam-x_cam,centere1_y_cam-y_cam]) * 2
            print ("legnth", length)
            print ("width", width)
            print ("angle",self.grasp_angle)
            print ("centre1_z:",centere1_z_cam," centre3_z:", centere3_z_cam)

            rack_pos.position.x = x_cam
            rack_pos.position.y = y_cam
            rack_pos.position.z = z_cam
            rack_pos.orientation.z = self.grasp_angle
            response.probablity    = 1.0

            if width>0.05 and width< 0.15:
                if not (centere1_z_cam <0.001 or  centere3_z_cam <0.001 or z_cam < 0.001): 
                    if length< 0.16:
                        if z_cam > 0.80: # lower row
                            # shift the centere (y dim) if neccessary
                            offset_x = 0*(0.2-length)*np.cos(self.grasp_angle*np.pi/180)
                            offset_y = 0*(0.2-length)*np.sin(self.grasp_angle*np.pi/180)
                            
                            print ("offsets:",offset_x,offset_y)
                            rack_pos.position.x -= offset_x
                            rack_pos.position.y -= offset_y

            if centere1_z_cam <0.001 or  centere3_z_cam <0.001 or z_cam < 0.001 : 
                response.probablity    = 0.0   
            elif length>0.22 or length< 0.08 :
                response.probablity    = 0.0 
            elif width>0.15 or width< 0.05 :
                response.probablity    = 0.0 
                
            
        else:
            rack_pos.position.x  = -1.0
            rack_pos.position.y  = -1.0
            rack_pos.position.z  = -1.0
            response.probablity  = 0.0
             

        response.pose = rack_pos        
        response.execute = 1 if (key== ord('e') or key ==ord('E')) else 0

        if response.execute :
            print (self.depth_array)


        
        # height, width = raw_image.shape[:2]

        # # Grasp point and color
        # grasp_point = [[response.pose.position.x, response.pose.position.y, response.pose.position.z]]
        # grasp_color = [[1.0, 0.0, 0.0]]

        # grasp_pcd = o3d.geometry.PointCloud()
        # grasp_pcd.points = o3d.utility.Vector3dVector(grasp_point)
        # grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_color)

        # # Using numpy to generate the point cloud
        # j, i = np.meshgrid(np.arange(width), np.arange(height))
        # z = raw_depth_image / 1000.0
        # x = (j - self.cx) * z / self.fx
        # y = (i - self.cy) * z / self.fy

        # raw_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        # rgb_points = raw_image.reshape(-1, 3) / 255.0

        # scene_pcd = o3d.geometry.PointCloud()
        # scene_pcd.points = o3d.utility.Vector3dVector(raw_points)
        # scene_pcd.colors = o3d.utility.Vector3dVector(rgb_points)

        # point = grasp_pcd.points[0]
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        # sphere.translate(point)
        # sphere.paint_uniform_color([1.0, 0.0, 0.0])

        # sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)

        # o3d.visualization.draw_geometries([scene_pcd + sphere_pcd])


        # # Extract the points within the ROI
        # roi_points = scene_pcd.select_by_indices(scene_pcd.crop([x, y, x + width, y + height]))
        # # Convert the ROI points to a numpy array
        # roi_points_np = np.asarray(roi_points.points)

        # # Fit a plane to the ROI points using RANSAC
        # plane_model, inliers = o3d.geometry.plane_from_points(
        #     roi_points_np, distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        
        # # Visualize the fitted plane
        # o3d.visualization.draw_geometries([roi_points, plane_model])


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

        # print ("camera", x_cam,y_cam,z_cam)
        
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

def main():

    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    
    main()
