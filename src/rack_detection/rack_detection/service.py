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
import open3d as o3d

import os
import datetime

import torch

from .InboxGraspPrediction import InboxGraspPrediction

# in my laptop we need to set this param to get it run on the GPU, not sure if we need this always.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" 

cv_bridge = CvBridge()

class MinimalService(Node):

    def __init__(self, pointcloud_vis=True):
        super().__init__('minimal_service')

        self.rgb_image   = None
        self.depth_image = None
        self.pointcloud_vis = pointcloud_vis

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
                self.grasp_angle = gs_list[0][3]
                

            print ("grasp list:\n", gs_list)
            plt.imshow(self.vision.image_rgb)
            plt.axis('on')
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

        print ("depth size:",self.depth_image.shape)
        print ("image size:",self.rgb_image.shape)
        
        # Extract depth value for the target pixel position
        if self.grasp_center is not None:
            depth_pixel_x = int(self.grasp_center[0])
            depth_pixel_y = int(self.grasp_center[1])

            # depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x][0])
            depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x])
            
            depth_in_meters = depth_value / 1000
            print (f"depth [{depth_pixel_x},{depth_pixel_y}] = {depth_value}")
            print (f"grasp angle: {self.grasp_angle}")
            #  float(frame[y_center][x_center])*100.0

            x_cam, y_cam, z_cam = self.pixel_to_meter(self.grasp_center[0],self.grasp_center[1],depth_in_meters)
            
            # rack_pos.position.x = float(self.grasp_center[0])
            # rack_pos.position.y = float(self.grasp_center[1])
            # rack_pos.position.z = depth_in_meters
            rack_pos.position.x = x_cam
            rack_pos.position.y = y_cam
            rack_pos.position.z = z_cam
            rack_pos.orientation.z = self.grasp_angle
            response.probablity  = 1.0

            if self.self.pointcloud_vis:
                print(f"Processing PointCloud visualization -> grasp position: ({x_cam}, {y_cam}, {z_cam})")
                height, weight = self.rgb_image.shape[:2]
                raw_points = []
                rgb_points = []

                grasp_point = [[x_cam, y_cam, z_cam]]
                grasp_color = [[1.0, 0.0, 0.0]]

                grasp_pcd = o3d.geometry.PointCloud()
                grasp_pcd.points = o3d.utility.Vector3dVector(grasp_point)
                grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_color)

                for i in range(height):
                    for j in range(weight):
                        z = self.depth_image[i][j] / 1000
                        x = (j - self.cx) * z / self.fx
                        y = (i - self.cy) * z / self.fy
                        a = 255
                        raw_points.append([x, y, z])
                        rgb_points.append(self.rgb_image[i][j] / 255)
                
                scene_pcd = o3d.geometry.PointCloud()
                scene_pcd.points = o3d.utility.Vector3dVector(raw_points)
                scene_pcd.colors = o3d.utility.Vector3dVector(rgb_points)

                
                
                point = grasp_pcd.points[0]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005) #create a small sphere to represent point
                sphere.translate(point) #translate this sphere to point
                sphere.paint_uniform_color([1.0, 0.0, 0.0])

                sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)
                
                o3d.visualization.draw_geometries([scene_pcd+sphere_pcd])


        else:
            rack_pos.position.x = -1.0
            rack_pos.position.y = -1.0
            rack_pos.position.z = -1.0
            response.probablity  = 0.0
             

        response.pose = rack_pos        
        response.execute = 1 if (key== ord('e') or key ==ord('E')) else 0

        if response.execute :
            print (self.depth_array)

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

def main():

    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    
    main()
