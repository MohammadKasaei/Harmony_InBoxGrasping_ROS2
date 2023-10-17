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
import open3d as o3d


from .GraspPrediction import GraspPrediction
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
        self.vision = GraspPrediction()
        self.srv = self.create_service(RackDetection, 'grasp_pose_detection', self.grasp_detection_callback)

    def camera_info_callback(self, msg: CameraInfo):
        # print ("camera:", msg.k)
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        # Use these parameters as needed

    def grasp_detection_callback(self, request, response):
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
        
        self.vision.image = raw_image.copy()      
        self.vision.depth = raw_depth_image.copy()      
        self.grasp_center = None

        grasps = self.vision.predict_grasp()
        for g in grasps:
            center_x, center_y, width, height, rad = g
            theta = rad / np.pi * 180
            box = ((self.vision.x_offset+center_x, self.vision.y_offset+center_y), (width, height), -(theta))
            box = cv2.boxPoints(box)
            box = np.int0(box)
            
            p1, p2, p3, p4 = box
            length = width
            p5 = (p1+p2)/2
            p6 = (p3+p4)/2
            p7 = (p5+p6)/2

            cv2.circle(self.vision.image, (int(p7[0]), int(p7[1])), 2, (0,0,255), 2)
            cv2.line(self.vision.image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 3, 8)
            cv2.line(self.vision.image, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 3, 8)
            cv2.line(self.vision.image, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (255,0,0), 2, 8)

            # grasp_msg.x = g[0]
            # grasp_msg.y = g[1]

            # grasp_msg.w = g[2]
            # grasp_msg.h = g[3]
            # grasp_msg.rot = g[4]
            
            print (g)
            print ("-"*40)
        cv2.imshow("RGB Image", self.vision.image)
        key = cv2.waitKey(0)

        # Extract depth value for the target pixel position
        if len(grasps)>0:
            center_x, center_y, width, height, rad = grasps[0]
            self.grasp_angle = rad*180/np.pi

            self.grasp_center = [self.vision.x_offset+center_x, self.vision.y_offset+center_y]

            depth_pixel_x = int(self.grasp_center[0])
            depth_pixel_y = int(self.grasp_center[1])

            depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x])
            
            depth_in_meters = depth_value / 1000
            print (f"depth [{depth_pixel_x},{depth_pixel_y}] = {depth_value}")
            print (f"grasp angle: {self.grasp_angle}")
            
            x_cam, y_cam, z_cam = self.pixel_to_meter(self.grasp_center[0],self.grasp_center[1],depth_in_meters)
            
            rack_pos.position.x = x_cam
            rack_pos.position.y = y_cam
            rack_pos.position.z = z_cam
            rack_pos.orientation.z = self.grasp_angle
            response.probablity    = 1.0
        else:
            rack_pos.position.x  = -1.0
            rack_pos.position.y  = -1.0
            rack_pos.position.z  = -1.0
            response.probablity  = 0.0
        
        response.pose = rack_pos        
        
        height, width = raw_image.shape[:2]
        # Grasp point and color
        grasp_point = [[response.pose.position.x, response.pose.position.y, response.pose.position.z]]
        grasp_color = [[1.0, 0.0, 0.0]]

        grasp_pcd = o3d.geometry.PointCloud()
        grasp_pcd.points = o3d.utility.Vector3dVector(grasp_point)
        grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_color)

        # Using numpy to generate the point cloud
        j, i = np.meshgrid(np.arange(width), np.arange(height))
        z = raw_depth_image / 1000.0
        x = (j - self.cx) * z / self.fx
        y = (i - self.cy) * z / self.fy

        raw_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        rgb_points = raw_image.reshape(-1, 3) / 255.0

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(raw_points)
        scene_pcd.colors = o3d.utility.Vector3dVector(rgb_points)

        point = grasp_pcd.points[0]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(point)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])

        sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)

        o3d.visualization.draw_geometries([scene_pcd + sphere_pcd])

        
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
