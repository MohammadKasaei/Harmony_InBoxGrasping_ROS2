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

import sys

# from example_interfaces.srv import AddTwoInts
from rack_detection_interface.srv import RackDetection

import rclpy
from rclpy.node import Node

cv_bridge = CvBridge()


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
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

        self.cli = self.create_client(RackDetection, 'rack_detection')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RackDetection.Request()


    def camera_info_callback(self, msg: CameraInfo):
        # print ("camera:", msg.k)
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        # Use these parameters as needed


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

    def send_request(self, dbg_vis, show_masks_point, show_raw_image):
        self.req.dbg_vis = dbg_vis
        self.req.show_masks_point = show_masks_point
        self.req.show_raw_image = show_raw_image
        
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()

    if len(sys.argv) > 3 :
        dbg_vis = int(sys.argv[1])
        show_masks_point = int(sys.argv[2])
        show_raw_image = int(sys.argv[3])
    else:
        dbg_vis = 0
        show_masks_point = 0
        show_raw_image   = 0
    

    response = minimal_client.send_request(dbg_vis, show_masks_point,show_raw_image)
    minimal_client.get_logger().info(
        'Params : %d , %d , %d response: pos:(%f,%f,%f), angle: %f, prob: %f, exec: %d' %
        (dbg_vis, show_masks_point,show_raw_image , \
         response.pose.position.x,response.pose.position.y,response.pose.position.z,
         response.pose.orientation.z,
         response.probablity,
         response.execute))
    
    depth_img = minimal_client.depth_image
    rgb_img   = minimal_client.rgb_image

    print(f"Processing PointCloud visualization -> grasp position: ({response.pose.position.x}, {response.pose.position.y}, {response.pose.position.z})")
    

    height, weight = minimal_client.rgb_image.shape[:2]
    raw_points = []
    rgb_points = []

    grasp_point = [[response.pose.position.x,response.pose.position.y,response.pose.position.z]]
    grasp_color = [[1.0, 0.0, 0.0]]

    grasp_pcd = o3d.geometry.PointCloud()
    grasp_pcd.points = o3d.utility.Vector3dVector(grasp_point)
    grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_color)

    for i in range(height):
        for j in range(weight):
            z = depth_img[i][j] / 1000
            x = (j - minimal_client.cx) * z / minimal_client.fx
            y = (i - minimal_client.cy) * z / minimal_client.fy            
            raw_points.append([x, y, z])
            rgb_points.append(rgb_img[i][j] / 255)
    
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(raw_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(rgb_points)
    
    point = grasp_pcd.points[0]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005) #create a small sphere to represent point
    sphere.translate(point) #translate this sphere to point
    sphere.paint_uniform_color([1.0, 0.0, 0.0])

    sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)
    
    o3d.visualization.draw_geometries([scene_pcd+sphere_pcd])

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

