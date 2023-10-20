import rclpy
from rclpy.node import Node
import math
from rack_detection_interface.srv import RackDetection
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import time
# import open3d as o3d

import os
import datetime

import torch

from .InboxGraspPrediction import InboxGraspPrediction
# in my laptop we need to set this param to get it run on the GPU, not sure if we need this always.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" 

cv_bridge = CvBridge()

def is_rotation_matrix(R):
    """Check if a matrix is a valid rotation matrix.

    Parameters
    ----------
    R : numpy.ndarray
        input matrix [R.shape : (3, 3)]

    Returns
    ------
    bool
        input matrix is a valid rotation matrix

    Raises
    ------
    ValueError
        If R is not a 3-by-3 matrix
    """
    R = np.array(R)
    if R.shape != (3,3):
        raise ValueError("R must be a 3-by-3 matrix")

    epsilon = 10**-3

    I = np.eye(3,3)
    if ( np.abs(np.linalg.det(R) - 1) > epsilon or 
        np.sum(I - np.matmul(R, R.T)) > epsilon ):
        return False
    else:
        return True
    
def rot_to_quat(R):
    """Rotation matrix to quaternion conversion.

    Parameters
    ----------
    R : numpy.ndarray
        rotation matrix [R.shape : (3, 3)]

    Returns
    ------
    quat: numpy.ndarray
        [qx, qy, qz, qw]

    Raises
    ------
    ValueError
        If R is not a valid rotation matrix
    """
    
    R = np.array(R)
    if not is_rotation_matrix(R):
        raise ValueError("R must be a valid rotation matrix")

    epsilon = 10**-5

    tr = np.trace(R)

    if tr > epsilon:

        sqrt_tr = np.sqrt(tr + 1.)
        qw = 0.5 * sqrt_tr
        qx = (R[2,1] - R[1,2]) / (2. * sqrt_tr)
        qy = (R[0,2] - R[2,0]) / (2. * sqrt_tr)
        qz = (R[1,0] - R[0,1]) / (2. * sqrt_tr)

    elif (R[1,1] > R[0,0]) and (R[1,1] > R[2,2]):

        # max value at R[1,1]
        sqrt_tr = np.sqrt(R[1,1] - R[0,0] - R[2,2] + 1.)

        qy = 0.5*sqrt_tr

        if sqrt_tr > epsilon:
            sqrt_tr = 0.5/sqrt_tr

        qw = (R[0,2] - R[2,0]) * sqrt_tr
        qx = (R[1,0] + R[0,1]) * sqrt_tr
        qz = (R[2,1] + R[1,2]) * sqrt_tr

    elif R[2,2] > R[0,0]:

        # max value at R[2,2]
        sqrt_tr = np.sqrt(R[2,2] - R[0,0] - R[1,1] + 1.)

        qz = 0.5*sqrt_tr

        if sqrt_tr > epsilon:
            sqrt_tr = 0.5/sqrt_tr

        qw = (R[1,0] - R[0,1]) * sqrt_tr
        qx = (R[0,2] + R[2,0]) * sqrt_tr
        qy = (R[2,1] + R[1,2]) * sqrt_tr

    else:

        # max value at R[0,0]
        sqrt_tr = np.sqrt(R[0,0] - R[1,1] - R[2,2] + 1.)

        qx = 0.5 * sqrt_tr

        if sqrt_tr > epsilon:
            sqrt_tr = 0.5 / sqrt_tr

        qw = (R[2,1] - R[1,2]) * sqrt_tr
        qy = (R[1,0] + R[0,1]) * sqrt_tr
        qz = (R[0,2] + R[2,0]) * sqrt_tr

    return np.array([qx, qy, qz, qw])


def quat_to_rot(quat):
    """Quaternion to rotation matrix conversion.

    Parameters
    ----------
    quat: numpy.ndarray
        [qx, qy, qz, qw]

    Returns
    ------
    R : numpy.ndarray
        rotation matrix [R.shape : (3, 3)]

    Raises
    ------
    ValueError
        If input list dimension is not 4
    """
    if len(quat) != 4:
        raise ValueError("Wrong input size")

    # check normalization
    quat = np.array(quat)
    norm = np.linalg.norm(quat)
    if  (1 - norm)**2 > 10**-4: # quat is not normalized
        quat = list(quat / np.linalg.norm(quat))

    qx, qy, qz, qw = quat

    R = np.zeros((3,3))
    R[0,0] = 1 - 2 * qy**2 - 2 * qz**2
    R[0,1] = 2 * qx * qy - 2 * qz * qw
    R[0,2] = 2 * qx * qz + 2 * qy * qw
    R[1,0] = 2 * qx * qy + 2 * qz * qw
    R[1,1] = 1 - 2 * qx**2 - 2 * qz**2
    R[1,2] = 2 * qy * qz - 2 * qx * qw
    R[2,0] = 2 * qx * qz - 2 * qy * qw
    R[2,1] = 2 * qy * qz + 2 * qx * qw
    R[2,2] = 1 - 2 * qx**2 - 2 * qy**2

    return R

def quaternion_from_euler(ai, aj, ak):
    """
    Covnert euler angles to quaternin

    Input
    :param ai: rotation around x
    :param aj: rotation around y
    :param ak: rotation around z

    Output
    :return: A 4 element array containing the final quaternion (qx, qy, qz, qw)

    """
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

def quaternion_multiply(q0, q1):
    """
    Multiplies two quaternions.

    Input
    :param q0: A 4 element array containing the first quaternion (qx, qy, qz, qw)
    :param q1: A 4 element array containing the second quaternion (qx, qy, qz, qw)

    Output
    :return: A 4 element array containing the final quaternion (qx, qy, qz, qw)

    """
    # Extract the values from q0
    x0 = q0[0]
    y0 = q0[1]
    z0 = q0[2]
    w0 = q0[3]

    # Extract the values from q1
    x1 = q1[0]
    y1 = q1[1]
    z1 = q1[2]
    w1 = q1[3]

    # Computer the product of the two quaternions, term by term
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([q0q1_x, q0q1_y, q0q1_z, q0q1_w])

    # Return a 4 element array containing the final quaternion
    return final_quaternion


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')

        print("Initializing rack detector node ...")
        self.flg_print_image_received = False

        self.rgb_image   = None
        self.depth_image = None

        self.cb_group_ = ReentrantCallbackGroup()

        rgb_subscription = self.create_subscription(
            CompressedImage,
            "/myumi_005/sensors/top_azure/rgb/image_raw/compressed",  
            self.rgb_image_callback,
            1,
            callback_group=self.cb_group_
        )

        depth_subscription = self.create_subscription(
            Image,
            "/myumi_005/sensors/top_azure/depth_to_rgb/image_raw",
            self.depth_image_callback,
            1,
            callback_group=self.cb_group_
        )

        camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/myumi_005/sensors/top_azure/rgb/camera_info',
            self.camera_info_callback,
            10,
            callback_group=self.cb_group_
        )

        torch.cuda.empty_cache()        
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        self.vision = InboxGraspPrediction()
        self.srv = self.create_service(
            RackDetection, 
            'rack_detection', 
            self.rack_detection_callback,
            callback_group=self.cb_group_
        )
        
        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize the transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.initialization_timer = self.create_timer(5.0, self.init)
        print(".........")

    def init(self):
        self.initialization_timer.cancel()
        try:
            rgb_to_base_tf = self.tf_buffer.lookup_transform(
                'myumi_005_top_azure_rgb_camera_link',
                'myumi_005_base_link',
                rclpy.time.Time())
            self.q_rgb_to_base = [
                rgb_to_base_tf.transform.rotation.x,
                rgb_to_base_tf.transform.rotation.y,
                rgb_to_base_tf.transform.rotation.z,
                rgb_to_base_tf.transform.rotation.w
            ]
        except TransformException as ex:
            self.get_logger().info(
                f'Could not get rgb-to-base transform: {ex}')
        self.q_rgb_to_base = [-0.684, 0.688, -0.185, -0.157] # qx, qy, qz, qw
            
        print("Rack detector service initialized")

    def get_last_transform(self, from_frame, to_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                rclpy.time.Time())
            return transform
        except TransformException as ex:
            self.get_logger().info(
                f'Could not get transform: {ex}')

    def camera_info_callback(self, msg: CameraInfo):
        # TODO: check timestamp
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

    def image_wait(self):
        print("Waiting for another image...")
        self.image_wait_timer.cancel()

    def rack_detection_callback(self, request, response):
        masks, scores = [],[]
        try_idx = 0
        while len (masks) == 0 and try_idx<1:

            print ("-"*40,try_idx)
            try_idx += 1

            rack_pos = Pose()
            if (self.rgb_image is None) or (self.depth_image is None):
                print("No images received so far!")
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

            # if len(masks) == 0:
            #     # self.flg_print_image_received = True
            #     self.image_wait_timer = self.create_timer(2.0, self.image_wait)
            # else:
            #     self.flg_print_image_received = False


        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # plt.subplot(121)
            # plt.imshow(self.vision.image_rgb)
            # plt.title('Original')
            # plt.axis('on')
            # plt.subplot(122)
            # # plt.imshow(gs.image)
            # plt.title('Grasp')
            # if request.show_masks_point or True:
            #     self.vision.show_mask(mask, plt.gca(),random_color=False)
            #     self.vision.show_points(self.vision._input_point, self.vision._input_label, plt.gca())
                
            gs_list = self.vision.generate_grasp(mask,vis=False)

            if len(gs_list)>0:
                self.grasp_center = gs_list[0][0]
                center1 = gs_list[0][1]        
                center2 = gs_list[0][2]        
                center3 = gs_list[0][3]        
                center4 = gs_list[0][4]        
                self.grasp_angle = gs_list[0][5]
                self.box = gs_list[0][6]
                print("angle: ", self.grasp_angle)
            
                centers = np.array([center1, center2, center3, center4])
                y_sorted_centers =np.array(sorted(centers, key=lambda a_entry: a_entry[1]))
                far_side_center = y_sorted_centers[0,:]
                

            # print ("grasp list:\n", gs_list)
            # key = cv2.waitKey(1)
            # plt.imshow(self.vision.image_rgb)
            # plt.axis('on')
            # plt.show()

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
        
        # Extract depth value for the target pixel position
        if self.grasp_center is not None:
            depth_pixel_x = int(self.grasp_center[0])
            depth_pixel_y = int(self.grasp_center[1])
            depth_value = float(raw_depth_image[depth_pixel_y,depth_pixel_x])
            depth_in_meters = depth_value / 1000
            x_cam, y_cam, z_cam = self.pixel_to_meter(self.grasp_center[0],self.grasp_center[1],depth_in_meters)
            idx = 0
            while z_cam < 0.001 and idx<15:
                depth_value = float(raw_depth_image[depth_pixel_y+idx,depth_pixel_x])
                depth_in_meters = depth_value / 1000
                x_cam, y_cam, z_cam = self.pixel_to_meter(self.grasp_center[0],self.grasp_center[1]+idx,depth_in_meters)
                idx+= 1
            print ("rgb_to_mask_center: ",x_cam, y_cam, z_cam)

            far_side_center_pixel_x = int(far_side_center[0])
            far_side_center_pixel_y = int(far_side_center[1])
            depth_value = float(raw_depth_image[far_side_center_pixel_y,far_side_center_pixel_x])
            depth_in_meters = depth_value / 1000
            far_side_center_x_cam, far_side_center_y_cam, far_side_center_z_cam = self.pixel_to_meter(far_side_center[0], far_side_center[1], depth_in_meters)
            idx = 0
            while far_side_center_z_cam < 0.001 and idx<20:
                depth_value = float(raw_depth_image[far_side_center_pixel_y+idx,far_side_center_pixel_x+int(idx/2)])
                depth_in_meters = depth_value / 1000
                far_side_center_x_cam, far_side_center_y_cam, far_side_center_z_cam = self.pixel_to_meter(far_side_center[0]+int(idx/2), far_side_center[1]+idx, depth_in_meters)
                idx+= 1

            print ("rgb_to_far_side_center: ", far_side_center_x_cam, far_side_center_y_cam, far_side_center_z_cam)

            # centere3_depth_pixel_x = int(center3[0])
            # centere3_depth_pixel_y = int(center3[1])
            # depth_value = float(raw_depth_image[centere3_depth_pixel_y,centere3_depth_pixel_x])
            # depth_in_meters = depth_value / 1000
            # centere3_x_cam, centere3_y_cam, centere3_z_cam = self.pixel_to_meter(center3[0]+10,center3[1],depth_in_meters)
            # idx = 0
            # while centere3_z_cam < 0.001 and idx<20:
            #     depth_value = float(raw_depth_image[centere3_depth_pixel_y+idx,centere3_depth_pixel_x+int(idx/2)])
            #     depth_in_meters = depth_value / 1000
            #     centere3_x_cam, centere3_y_cam, centere3_z_cam = self.pixel_to_meter(center3[0]+10+int(idx/2),center3[1]+idx,depth_in_meters)
            #     idx+= 1
            
            # centere1_depth_pixel_x = int(center1[0]+10)
            # centere1_depth_pixel_y = int(center1[1])
            # depth_value = float(raw_depth_image[centere1_depth_pixel_y,centere1_depth_pixel_x])
            # depth_in_meters = depth_vamaskslue / 1000
            # centere1_x_cam, centere1_y_cam, centere1_z_cam = self.pixel_to_meter(center1[0],center1[1],depth_in_meters)
            # idx = 0
            # while centere1_z_cam < 0.001 and idx<15:
            #     depth_value = float(raw_depth_image[centere1_depth_pixel_y+idx,centere1_depth_pixel_x])
            #     depth_in_meters = depth_value / 1000
            #     centere1_x_cam, centere1_y_cam, centere1_z_cam = self.pixel_to_meter(center1[0],center1[1]+idx,depth_in_meters)
            #     idx+= 1

            # length = np.linalg.norm ([centere3_x_cam-x_cam,centere3_y_cam-y_cam]) * 2
            # width = np.linalg.norm ([centere1_x_cam-x_cam,centere1_y_cam-y_cam]) * 2
            # print ("length", length)
            # print ("width", width)
            # print ("angle",self.grasp_angle)
            # print ("centre1_z:",centere1_z_cam," centre3_z:",centere3_z_cam)

            if far_side_center_z_cam < 0.20: # high row (original_threshold=0.79, threshold reduced to go always on the else case) TODO

                rack_pos.position.x = x_cam
                rack_pos.position.y = y_cam
                rack_pos.position.z = z_cam
                q_base_to_rack = quaternion_from_euler(0, 0, self.grasp_angle * np.pi / 180)
                q_rgb_to_rack = quaternion_multiply(self.q_rgb_to_base, q_base_to_rack) 
                rack_pos.orientation.x = q_rgb_to_rack[0]
                rack_pos.orientation.y = q_rgb_to_rack[1]
                rack_pos.orientation.z = q_rgb_to_rack[2]
                rack_pos.orientation.w = q_rgb_to_rack[3]
                response.probablity    = 1.0

                # offset_x = 0.0
                # offset_y = 0.0
                # if width>0.05 and width< 0.15:
                #     if not (centere1_z_cam <0.001 or  centere3_z_cam <0.001 or z_cam < 0.001): 
                #         if length< 0.16:
                #             if z_cam > 0.80: # lower row
                #                 # shift the centere (y dim) if neccessary
                #                 offset_x = 0.5*(0.2-length)*np.cos(self.grasp_angle*np.pi/180)
                #                 offset_y = 0.5*(0.2-length)*np.sin(self.grasp_angle*np.pi/180)
                #                 print ("offsets:",offset_x,offset_y)


                # if centere1_z_cam <0.001 or  centere3_z_cam <0.001 or z_cam < 0.001 : 
                #     response.probablity    = 0.0   
                # elif width>0.15 or width< 0.05 :
                #     response.probablity    = 0.0   
                # elif length>0.23 or length< 0.08 :
                #     response.probablity    = 0.0   

                # convert rgb-ro-rack transform to homogenous matrix
                H_rgb_to_rack = np.eye(4)
                q_rgb_to_rack = np.array(
                    [
                        rack_pos.orientation.x,
                        rack_pos.orientation.y,
                        rack_pos.orientation.z,
                        rack_pos.orientation.w
                    ]
                )
                p_rgb_to_rack = np.array(
                    [
                        rack_pos.position.x,
                        rack_pos.position.y,
                        rack_pos.position.z,
                    ]
                )
                H_rgb_to_rack[:3,:3] = quat_to_rot(q_rgb_to_rack)
                H_rgb_to_rack[:3,3] = p_rgb_to_rack  
            
            else: # lowest row

                # get rack center from the far rack side center point applying an offset
                p_rgb_to_rack_far_side = np.array([far_side_center_x_cam, far_side_center_y_cam, far_side_center_z_cam])
                q_base_to_rack_far_side = quaternion_from_euler(0, 0, self.grasp_angle * np.pi / 180)
                q_base_to_rack_far_side = quaternion_multiply(self.q_rgb_to_base, q_base_to_rack_far_side) 
                H_rgb_to_rack_far_side = np.eye(4)
                H_rgb_to_rack_far_side[:3,:3] = quat_to_rot(q_base_to_rack_far_side)
                H_rgb_to_rack_far_side[:3,3] = p_rgb_to_rack_far_side

                H_offset = np.eye(4)
                H_offset[0,3] = -0.10 # half the long side of the rack

                H_rgb_to_rack = np.matmul(H_rgb_to_rack_far_side, H_offset)
                q_rgb_to_rack = rot_to_quat(H_rgb_to_rack[:3,:3])
                p_rgb_to_rack = H_rgb_to_rack[:3,3]

                rack_pos.position.x = p_rgb_to_rack[0]
                rack_pos.position.y = p_rgb_to_rack[1]
                rack_pos.position.z = p_rgb_to_rack[2]
                rack_pos.orientation.x = q_rgb_to_rack[0]
                rack_pos.orientation.y = q_rgb_to_rack[1]
                rack_pos.orientation.z = q_rgb_to_rack[2]
                rack_pos.orientation.w = q_rgb_to_rack[3]
                response.probablity    = 1.0

                # tf_db_far_side = TransformStamped()
                # tf_db_far_side.header.stamp = self.get_clock().now().to_msg()
                # tf_db_far_side.header.frame_id = "myumi_005_top_azure_rgb_camera_link"
                # tf_db_far_side.child_frame_id = "rack_far_side"
                # tf_db_far_side.transform.translation.x = p_rgb_to_rack_far_side[0]
                # tf_db_far_side.transform.translation.y = p_rgb_to_rack_far_side[1]
                # tf_db_far_side.transform.translation.z = p_rgb_to_rack_far_side[2]
                # tf_db_far_side.transform.rotation.x = q_base_to_rack_far_side[0]
                # tf_db_far_side.transform.rotation.y = q_base_to_rack_far_side[1]
                # tf_db_far_side.transform.rotation.z = q_base_to_rack_far_side[2]
                # tf_db_far_side.transform.rotation.w = q_base_to_rack_far_side[3]
                # self.tf_broadcaster.sendTransform(tf_db_far_side)

                # tf_db_center_original = TransformStamped()
                # tf_db_center_original.header.stamp = self.get_clock().now().to_msg()
                # tf_db_center_original.header.frame_id = "myumi_005_top_azure_rgb_camera_link"
                # tf_db_center_original.child_frame_id = "rack_center_original"
                # tf_db_center_original.transform.translation.x = x_cam
                # tf_db_center_original.transform.translation.y = y_cam
                # tf_db_center_original.transform.translation.z = z_cam
                # tf_db_center_original.transform.rotation.x = q_base_to_rack_far_side[0]
                # tf_db_center_original.transform.rotation.y = q_base_to_rack_far_side[1]
                # tf_db_center_original.transform.rotation.z = q_base_to_rack_far_side[2]
                # tf_db_center_original.transform.rotation.w = q_base_to_rack_far_side[3]
                # self.tf_broadcaster.sendTransform(tf_db_center_original)
            
        else:
            rack_pos.position.x  = -1.0
            rack_pos.position.y  = -1.0
            rack_pos.position.z  = -1.0
            response.probablity  = 0.0

        response.pose = rack_pos    
        # response.execute = 1 if (key== ord('e') or key ==ord('E')) else 0

        if response.execute :
            print (self.depth_array)

        torch.cuda.empty_cache()
        
        # Publish tf w.r.t. APRIL 2 (marker on wall)
        if response.probablity  != 0.0:
            # get wall-to-rgb transform
            wall_to_rgb = self.get_last_transform("april_2", "myumi_005_top_azure_rgb_camera_link") # TODO: handle error if april_2 not available
            
            # convert wall-to-rgb transform to homogenous matrix
            H_wall_to_rgb = np.eye(4)
            q_wall_to_rgb = np.array(
                [
                    wall_to_rgb.transform.rotation.x,
                    wall_to_rgb.transform.rotation.y,
                    wall_to_rgb.transform.rotation.z,
                    wall_to_rgb.transform.rotation.w
                ]
            )
            p_wall_to_rgb = np.array(
                [
                    wall_to_rgb.transform.translation.x,
                    wall_to_rgb.transform.translation.y,
                    wall_to_rgb.transform.translation.z,
                ]
            )
            H_wall_to_rgb[:3,:3] = quat_to_rot(q_wall_to_rgb)
            H_wall_to_rgb[:3,3] = p_wall_to_rgb

            # print("H_rgb_to_rack", H_rgb_to_rack)
            # print("H_wall_to_rgb", H_wall_to_rgb)

            H_wall_to_rack = np.matmul(H_wall_to_rgb, H_rgb_to_rack)
            # print("H_wall_to_rack", H_wall_to_rack)
            
            q_wall_to_rack = rot_to_quat(H_wall_to_rack[:3,:3])
            p_wall_to_rack = H_wall_to_rack[:3,3]

            self.wall_to_rack_tf_stay_alive_count = 0
            self.tf_msg_wall_to_rack = TransformStamped()
            self.tf_msg_wall_to_rack.header.stamp = self.get_clock().now().to_msg()
            self.tf_msg_wall_to_rack.header.frame_id = "april_2"
            self.tf_msg_wall_to_rack.child_frame_id = "rack_for_dock"
            self.tf_msg_wall_to_rack.transform.translation.x = p_wall_to_rack[0]
            self.tf_msg_wall_to_rack.transform.translation.y = p_wall_to_rack[1]
            self.tf_msg_wall_to_rack.transform.translation.z = p_wall_to_rack[2]
            self.tf_msg_wall_to_rack.transform.rotation.x = q_wall_to_rack[0]
            self.tf_msg_wall_to_rack.transform.rotation.y = q_wall_to_rack[1]
            self.tf_msg_wall_to_rack.transform.rotation.z = q_wall_to_rack[2]
            self.tf_msg_wall_to_rack.transform.rotation.w = q_wall_to_rack[3]
            self.tf_broadcaster.sendTransform(self.tf_msg_wall_to_rack)
            print("Publishing rack pose w.r.t. april_2 marker (wall)")

            self.tf_msg_rgb_to_rack = TransformStamped()
            self.tf_msg_rgb_to_rack.header.stamp = self.get_clock().now().to_msg()
            self.tf_msg_rgb_to_rack.header.frame_id = "myumi_005_top_azure_rgb_camera_link"
            self.tf_msg_rgb_to_rack.child_frame_id = "rack_for_grasp"
            self.tf_msg_rgb_to_rack.transform.translation.x = p_rgb_to_rack[0]
            self.tf_msg_rgb_to_rack.transform.translation.y = p_rgb_to_rack[1]
            self.tf_msg_rgb_to_rack.transform.translation.z = p_rgb_to_rack[2]
            self.tf_msg_rgb_to_rack.transform.rotation.x = q_rgb_to_rack[0]
            self.tf_msg_rgb_to_rack.transform.rotation.y = q_rgb_to_rack[1]
            self.tf_msg_rgb_to_rack.transform.rotation.z = q_rgb_to_rack[2]
            self.tf_msg_rgb_to_rack.transform.rotation.w = q_rgb_to_rack[3]
            self.tf_broadcaster.sendTransform(self.tf_msg_rgb_to_rack)
            print("Publishing rack pose w.r.t. rgb_camera_link")

            # wall_to_rack TF stay-alive timer 
            self.wall_to_rack_tf_stay_alive_count = 0
            # Call on_timer function every second
            self.wall_to_rack_tf_stay_alive_timer = self.create_timer(
                0.1, 
                self.on_wall_to_rack_tf_stay_alive_timer, 
            )

        return response
    

    def on_wall_to_rack_tf_stay_alive_timer(self):
        if self.tf_msg_wall_to_rack is not None:
            # Publish rgb to rack tf if available
            if self.wall_to_rack_tf_stay_alive_count >= 3600: # TODO: parametrize
                self.tf_msg_wall_to_rack = None # rgb to racl tf msg
                self.wall_to_rack_tf_stay_alive_timer.cancel()
                print("Stop re-publishing rack pose w.r.t. april_2 marker (wall)") # TODO: bug it keeps entering here
            else:
                # update timestamp and send again
                self.tf_msg_wall_to_rack.header.stamp = self.get_clock().now().to_msg()
                self.tf_msg_wall_to_rack.header.stamp = self.get_clock().now().to_msg()
                self.tf_broadcaster.sendTransform(self.tf_msg_wall_to_rack)
                self.wall_to_rack_tf_stay_alive_count += 1

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
        
        return x_cam, y_cam, z_cam


    def rgb_image_callback(self,msg):
        if self.flg_print_image_received:
            print("RGB image recevied!!!")
        # TODO: check timestamp
        try:
            self.rgb_image = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"Error: {str(e)}")

    def depth_image_callback(self,msg):
        if self.flg_print_image_received:
            print("Depth image recevied!!!")
        # TODO: check timestamp
        try:
            self.depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")          
        except Exception as e:
            print(f"Error: {str(e)}")

def main():

    rclpy.init()
    minimal_service = MinimalService()
    executor = MultiThreadedExecutor()
    executor.add_node(minimal_service)
    try:
        minimal_service.get_logger().info("Starting server node, shut down with CTRL-C")
        executor.spin()
    except KeyboardInterrupt:
        minimal_service.get_logger().info('Keyboard interrupt, shutting down.\n')
    minimal_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    
    main()
