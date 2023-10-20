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
    

    # height, weight = minimal_client.rgb_image.shape[:2]
    # raw_points = []
    # rgb_points = []

    # grasp_point = [[response.pose.position.x,response.pose.position.y,response.pose.position.z]]
    # grasp_color = [[1.0, 0.0, 0.0]]

    # grasp_pcd = o3d.geometry.PointCloud()
    # grasp_pcd.points = o3d.utility.Vector3dVector(grasp_point)
    # grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_color)

    # for i in range(height):
    #     for j in range(weight):
    #         z = depth_img[i][j] / 1000
    #         x = (j - minimal_client.cx) * z / minimal_client.fx
    #         y = (i - minimal_client.cy) * z / minimal_client.fy            
    #         raw_points.append([x, y, z])
    #         rgb_points.append(rgb_img[i][j] / 255)
    
    # scene_pcd = o3d.geometry.PointCloud()
    # scene_pcd.points = o3d.utility.Vector3dVector(raw_points)
    # scene_pcd.colors = o3d.utility.Vector3dVector(rgb_points)
    
    # point = grasp_pcd.points[0]
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005) #create a small sphere to represent point
    # sphere.translate(point) #translate this sphere to point
    # sphere.paint_uniform_color([1.0, 0.0, 0.0])

    # sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)
    
    # o3d.visualization.draw_geometries([scene_pcd+sphere_pcd])

    # import numpy as np
    # import open3d as o3d

    height, width = minimal_client.rgb_image.shape[:2]

    # Grasp point and color
    grasp_point = [[response.pose.position.x, response.pose.position.y, response.pose.position.z]]
    grasp_color = [[1.0, 0.0, 0.0]]

    grasp_pcd = o3d.geometry.PointCloud()
    grasp_pcd.points = o3d.utility.Vector3dVector(grasp_point)
    grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_color)

    # Using numpy to generate the point cloud
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_img / 1000.0
    x = (j - minimal_client.cx) * z / minimal_client.fx
    y = (i - minimal_client.cy) * z / minimal_client.fy

    raw_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgb_points = rgb_img.reshape(-1, 3) / 255.0

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(raw_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(rgb_points)

    point = grasp_pcd.points[0]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.translate(point)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])

    sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)

    o3d.visualization.draw_geometries([scene_pcd + sphere_pcd])


    

    # Define the parameters of the OrientedBoundingBox
    center = np.array([response.pose.position.x, response.pose.position.y, response.pose.position.z])  # Center of the box
    extent = np.array([0.1, 0.2, 0.05])    # Half extents of the box
    rotation = np.identity(3)               # Rotation matrix (identity for no rotation)

    # Create the OrientedBoundingBox
    obb = o3d.geometry.OrientedBoundingBox(center=center, extent=extent, R=rotation)

    # Crop the point cloud using the OrientedBoundingBox
    roi_points = scene_pcd.crop(obb)
    print ("--"*40,roi_points)
    o3d.visualization.draw_geometries([roi_points + sphere_pcd])
    pcd = roi_points

    
    # Estimate normals
    o3d.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply RANSAC to find the plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)

    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # If you want to visualize the inliers and outliers
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # Visualize
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])






    # Fit a plane to the ROI points using RANSAC
    plane_model, inliers = o3d.geometry.plane_from_points(
        np.asarray(roi_points.points), distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )

    # Get the normal vector of the fitted plane
    plane_normal = plane_model.normal

    # # Calculate the angles (in degrees) of the normal vector
    # roll_angle = np.arctan2(plane_normal[1], plane_normal[2]) * 180 / np.pi
    # pitch_angle = np.arctan2(-plane_normal[0], np.sqrt(plane_normal[1]**2 + plane_normal[2]**2)) * 180 / np.pi
    # yaw_angle = 0  # Assuming the plane's yaw angle is 0 since it's a 2D plane

    # # Print the angles
    # print("Roll Angle (degrees):", roll_angle)
    # print("Pitch Angle (degrees):", pitch_angle)
    # print("Yaw Angle (degrees):", yaw_angle)

    # Visualize the point cloud and the fitted plane (optional)
    o3d.visualization.draw_geometries([scene_pcd, roi_points, plane_model])

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

