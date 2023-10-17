import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import torch
from .grconvnet3 import GenerativeResnet
from skimage.filters import gaussian
from skimage.feature import peak_local_max

import os

class GraspPrediction():

    def __init__(self, device = "cuda" ) -> None:
        
        self._device = device
        self._visualize = True
        
        
        self.net = GenerativeResnet(
            input_channels=4,
            dropout=True,
            prob=0.1,
            channel_size=32
        ).to(device)

        self.net.load_state_dict(torch.load("models/Gr-ConvNet-cornell.pth"))
        self.bridge = CvBridge()


        self.net.eval().cuda()

        self.image = None
        self.depth = None

    
    def normalize_and_to_rgb(self, img):
        img = img.astype(np.float32) / 255.0
        img -= img.mean()
        img = img[:, :, (2, 1, 0)]
        return img
    
    
    def normalize_depth(self, img):
        img = np.clip((img - img.mean()), -1, 1)

        return img
    

    def depth_inpaint(self, depth_img, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(depth_img).max()
        depth_img = depth_img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_img = depth_img[1:-1, 1:-1]
        depth_img = depth_img * scale

        return depth_img

    
    def pad_to_square(self, rgb, depth):
        h, w, c = rgb.shape
        pad_size = max(h, w)
        pad_img = np.zeros((pad_size, pad_size, c), dtype='float32')
        pad_img[:, :, :] = self.norm_mean
        pad_depth = np.zeros((pad_size, pad_size), dtype='float32')

        pad_img[0: h, 0: w, :] = rgb
        pad_depth[0: h, 0: w] = depth

        # pad_img = cv2.resize(pad_img, (self.input_size, self.input_size))
        # pad_depth = cv2.resize(pad_depth, (self.input_size, self.input_size))

        return pad_img, pad_depth
    

    def post_process_output(self, q_img, cos_img, sin_img, width_img, num_grasps):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * 75.0

        q_img = gaussian(q_img, 2.0, preserve_range=True)
        ang_img = gaussian(ang_img, 2.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=num_grasps)

        grasps = []
        for grasp_point_array in local_max:
            grasp_point = tuple(grasp_point_array)

            grasp_angle = ang_img[grasp_point]

            grasp_width = width_img[grasp_point]
            grasp_height = 50
            
            grasps.append([int(grasp_point[1]), int(grasp_point[0]), int(grasp_width), int(grasp_height), grasp_angle])
        
        return grasps

    
    def pixel_to_ee(self, x, y):
        # pc_msg = rospy.wait_for_message('/points', PointCloud2, timeout=5)
        # pc_np = np.asarray(list(point_cloud2.read_points(pc_msg))).reshape(self.img_height, self.img_width, 3)
        
        # point_3d_camera_frame = pc_np[720-y,1280-x]

        # grasp_point_cam = PointStamped()
        # grasp_point_cam.header.frame_id = "kinect_subordinate_rgb_camera_link"
        # grasp_point_cam.header.stamp = rospy.Time(0)
        # grasp_point_cam.point.x = point_3d_camera_frame[0]
        # grasp_point_cam.point.y = point_3d_camera_frame[1]
        # grasp_point_cam.point.z = point_3d_camera_frame[2]

        # grasp_point_base = self.listener.transformPoint("/myumi_005_base_link", grasp_point_cam)

        # return grasp_point_base
        return None
    

    def predict_grasp(self,vis_masks = False,vis_output=False,color_picker = False):
        # Define the region of interest (ROI) coordinates
        self.x_offset = 200  # starting x-coordinate
        self.y_offset = 10  # starting y-coordinate
        self.width    = 700  # width of the ROI
        self.height   = 350  # height of the ROI
        
        # Crop the image using numpy array slicing
        image = self.image[self.y_offset:self.y_offset +self.height, self.x_offset :self.x_offset +self.width].copy()
        depth = self.depth[self.y_offset:self.y_offset +self.height, self.x_offset :self.x_offset +self.width].copy()
        if vis_masks:
            cv2.imshow('image', image)
            # cv2.waitKey(0)

        depth_image = depth
        depth_image = self.depth_inpaint(depth_image)
        
        rgb_image = image
        h, w, c = rgb_image.shape

        # rgb, depth = self.pad_to_square(rgb_image, depth_image)
        rgb_norm = self.normalize_and_to_rgb(rgb_image)
        depth_norm = self.normalize_depth(depth_image)

        # print(rgb_norm.shape, depth_norm.shape)

        if len(depth_norm.shape) < 3:
            depth_norm = np.expand_dims(depth_norm, -1)
        
        input_tensor = torch.from_numpy(np.concatenate([rgb_norm, depth_norm], axis=-1)).unsqueeze(0).permute(0, 3, 1, 2).float().to(self._device)

        # ==========================================
        # Process input with network
        # ==========================================
        with torch.no_grad():
            pos_output, cos_output, sin_output, width_output = self.net(input_tensor)
        
        grasps = self.post_process_output(pos_output, cos_output, sin_output, width_output, 10)
        for i in range(len(grasps)):
            g = grasps[i]
            # grasp_msg.x = g[0]
            # grasp_msg.y = g[1]
            # grasp_msg.w = g[2]
            # grasp_msg.h = g[3]
            # grasp_msg.rot = g[4]
            
            print (g)
            print ("-"*40)
            
        print("Finised detecting grasps")

        if self._visualize:
            fig = plt.figure(figsize=(10, 10))

            plt.ion()
            plt.clf()
            ax = fig.add_subplot(2, 3, 1)
            ax.imshow(rgb_image)
            ax.set_title('RGB')
            ax.axis('off')

            ax = fig.add_subplot(2, 3, 2)
            ax.imshow(depth_image, cmap='gray')
            ax.set_title('Depth')
            ax.axis('off')

            ax = fig.add_subplot(2, 3, 3)
            plot = ax.imshow(pos_output.cpu().numpy().squeeze(), cmap='jet', vmin=0, vmax=1)
            ax.set_title('Quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 4)
            ang_img = (torch.atan2(sin_output, cos_output) / 2.0).cpu().numpy().squeeze()
            plot = ax.imshow(ang_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
            ax.set_title('Angle')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 5)
            plot = ax.imshow(width_output.cpu().numpy().squeeze(), cmap='jet', vmin=0, vmax=1)
            ax.set_title('Width')
            ax.axis('off')
            plt.colorbar(plot)

            for g in grasps:
                center_x, center_y, self.width, self.height, rad = g
                theta = rad / np.pi * 180
                box = ((center_x, center_y), (self.width, self.height), -(theta))
                box = cv2.boxPoints(box)
                box = np.int0(box)
                # cv2.drawContours(img_fused, [box], 0, color, 2)

                p1, p2, p3, p4 = box
                length = self.width
                p5 = (p1+p2)/2
                p6 = (p3+p4)/2
                p7 = (p5+p6)/2

                cv2.circle(rgb_image, (int(p7[0]), int(p7[1])), 2, (0,0,255), 2)
                cv2.line(rgb_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 3, 8)
                cv2.line(rgb_image, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 3, 8)
                cv2.line(rgb_image, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (255,0,0), 2, 8)

            ax = fig.add_subplot(2, 3, 6)
            ax.imshow(rgb_image)
            ax.set_title('RGB')
            ax.axis('off')

            fig.savefig('results.png')

            fig.canvas.draw()
            plt.show()
            # plt.close(fig)
        
        return grasps
    
        
    def generate_grasp(self,mask,vis=True):

        (contours, hierarchy) = cv2.findContours(np.uint8(mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 3 
        j = 0
        grasp_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500 and area < 27000:
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

        # if grasp_list == [] :
        for i, contour in enumerate (contours):
            area = cv2.contourArea(contour) 
            print (f"contour area {i}: {area}")
        

        return grasp_list

