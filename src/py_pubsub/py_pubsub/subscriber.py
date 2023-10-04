#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime



class InBoxGraspingPipeline():
    def __init__(self) -> None:
        pass

    def pick_color(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = self.hsv_image[y,x]

            # HSV values
            hue = pixel[0]
            saturation = pixel[1]
            value = pixel[2]
            
            print(f"HSV Value at ({x},{y}): H:{hue}, S:{saturation}, V:{value}")

    def find_box_centre_depth(self,image,vis_masks = False,vis_output=False,color_picker = False):
        # Define the region of interest (ROI) coordinates
        x = 40  # starting x-coordinate
        y = 20  # starting y-coordinate
        width  = 280  # width of the ROI
        height = 140  # height of the ROI

        # Crop the image using numpy array slicing
        image = image[y:y+height, x:x+width]
        if vis_masks:
            cv2.imshow('image', image)


        # # hsv filter
        # self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # h_min, h_max = 10 , 200
        # s_min, s_max = 10 , 100
        # v_min, v_max = 30,  140

        # lower_threshold = np.array([h_min, s_min, v_min])
        # upper_threshold = np.array([h_max, s_max, v_max])
        # if color_picker:
        #     cv2.imshow('hsv_image', self.hsv_image)
        #     cv2.setMouseCallback('hsv_image', self.pick_color)

        # mask = cv2.inRange(self.hsv_image, lower_threshold, upper_threshold)
        # if vis_masks:
        #     cv2.imshow('mask', mask)

        # result = cv2.bitwise_and(image, image, mask=mask)

        # Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(image, (3, 3), 5)
        # if vis_masks:
        #     cv2.imshow('blurred', blurred)

        # Apply Canny edge detection
        # edges = cv2.Canny(blurred, 200, 255)
        edges = cv2.Canny(image, 220, 255)        
        if vis_masks:
            cv2.imshow('d_edge', edges)
            # cv2.waitKey(0)


        # Define the structuring element for erosion
        # Here, we're using a 5x5 rectangle
        # kernel = np.ones((3,3), np.uint8)
        # # Apply erosion
        # eroded_image = cv2.erode(edges, kernel, iterations=2)
        # if vis_output:
        #     cv2.imshow('eroded', eroded_image)

        # Define the kernel for dilation        
        kernel = np.ones((3, 3), np.uint8)  # Adjust the kernel size according to your needs
        # Perform dilation
        dilated = cv2.dilate(edges, kernel, iterations=1)
        if vis_output:
            cv2.imshow('dilated', dilated)

        # Find contours in the mask
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Define a perimeter threshold
        # min_perimeter = 100
        # max_perimeter = 500

        # filtered_contours = []
        # for contour in contours:
        #     perimeter = cv2.arcLength(contour, True)
        #     x, y, w, h = cv2.boundingRect(contour)
        #     aspect_ratio = float(w)/h

        #     # Filtering conditions
        #     # if min_perimeter < perimeter < max_perimeter and 0.9 < aspect_ratio < 1.1:  # Adjust as needed
        #     if  0.8 < aspect_ratio < 1.2:  # Adjust as needed
            
        #         filtered_contours.append(contour)



        # Detect contours with hierarchy
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Extract external contours based on hierarchy information
        external_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]


        # # Define a perimeter threshold
        # min_perimeter = 100
        # max_perimeter = 500

        filtered_contours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            # Filtering conditions
            # if min_perimeter < perimeter < max_perimeter and 0.9 < aspect_ratio < 1.1:  # Adjust as needed
            if  0.7 < aspect_ratio < 1.3:  # Adjust as needed
                filtered_contours.append(contour)


        # # Define a size threshold (in terms of contour area)
        min_contour_area = 1000  # Example value, adjust as needed
        max_contour_area = 4000  # Example value, adjust as needed
        # Filter contours by size
        filtered_contours2 = [contour for contour in filtered_contours if min_contour_area < cv2.contourArea(contour) < max_contour_area]


        if filtered_contours2 != [] :
            # Find the largest contour by area
            largest_contour = max(filtered_contours2, key=cv2.contourArea)

            # Draw the largest contour on a copy of the original image
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored drawing

            cv2.drawContours(output, [largest_contour], 0, (0, 0, 255), 2)  # Drawing in green color
            cv2.drawContours(output, filtered_contours2, -1, (0, 255, 0), 1)  # Drawing all contours in green color


            if vis_output:
                cv2.imshow('conrours', output)

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
                cv2.waitKey(10)
                # cv2.destroyAllWindows()
            return [x+center_x, y+center_y]


    def find_box_centre(self,image,vis_masks = False,vis_output=False,color_picker = False):
        # Define the region of interest (ROI) coordinates
        x = 200  # starting x-coordinate
        y = 100  # starting y-coordinate
        width  = 700  # width of the ROI
        height = 350  # height of the ROI

        # Crop the image using numpy array slicing
        image = image[y:y+height, x:x+width].copy()
        if vis_masks:
            cv2.imshow('image', image)


        # # hsv filter
        # self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # h_min, h_max = 10 , 200
        # s_min, s_max = 10 , 100
        # v_min, v_max = 30,  140

        # lower_threshold = np.array([h_min, s_min, v_min])
        # upper_threshold = np.array([h_max, s_max, v_max])
        # if color_picker:
        #     cv2.imshow('hsv_image', self.hsv_image)
        #     cv2.setMouseCallback('hsv_image', self.pick_color)

        # mask = cv2.inRange(self.hsv_image, lower_threshold, upper_threshold)
        # if vis_masks:
        #     cv2.imshow('mask', mask)

        # result = cv2.bitwise_and(image, image, mask=mask)

        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(result, (3, 3), 0)
        # if vis_masks:
        #     cv2.imshow('blurred', blurred)


        # # Define the sharpening kernel
        # sharpening_kernel = np.array([[-1, -1, -1],
        #                             [-1,  9, -1],
        #                             [-1, -1, -1]])

        # # Apply the kernel to the image using cv2.filter2D function
        # sharpened_image = cv2.filter2D(blurred, -1, sharpening_kernel)

        # Apply Canny edge detection
        # edges = cv2.Canny(blurred, 100, 200)
        # edges = cv2.Canny(result, 200, 255)   
        result = cv2.Canny(blurred, 50, 150)   
        
        if vis_masks:
            cv2.imshow('edge', result)
            # cv2.waitKey(0)

        # Define the kernel for dilation
        kernel = np.ones((7, 7), np.uint8)  # Adjust the kernel size according to your needs
        # Perform dilation
        dilated = cv2.dilate(result, kernel, iterations=2)
        if vis_output:
            cv2.imshow('dilated', dilated)


        # Detect contours with hierarchy
        # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Extract external contours based on hierarchy information
        # external_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]
        

        # Define the structuring element for erosion
        # Here, we're using a 5x5 rectangle
        kernel = np.ones((3,3), np.uint8)

        # Apply erosion
        eroded_image = cv2.erode(dilated, kernel, iterations=8)
        if vis_output:
            cv2.imshow('eroded', eroded_image)

        #  Find contours in the mask
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, contours, -1, (255, 0, 0), 3)  # Drawing all contours in green color


        # # Define a perimeter threshold
        # min_perimeter = 100
        # # max_perimeter = 500

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
        min_contour_area = 15000  # Example value, adjust as needed
        max_contour_area = 80000  # Example value, adjust as needed
        # Filter contours by size
        filtered_contours2 = [contour for contour in filtered_contours if min_contour_area < cv2.contourArea(contour) < max_contour_area]
        # filtered_contours2= []


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
                cv2.waitKey(10)
                # cv2.destroyAllWindows()
            return [x+center_x, y+center_y]

    




vision = InBoxGraspingPipeline()

def rgb_image_callback(msg):
    cv_bridge = CvBridge()
  
    try:
        rgb_image = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # print (rgb_image.shape)
        vision.find_box_centre (rgb_image,vis_output=True,vis_masks=True,color_picker=True)

        cv2.imshow("RGB Image", rgb_image)
        key = cv2.waitKey(10)  # Adjust the delay as needed (e.g., 10 milliseconds)
        if key == ord('s') or key ==ord('S'):
            if not os.path.isdir('capture_images'):
                os.makedirs('capture_images')
                print ("directory is creasted.")
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"capture_images/{current_time}_captured.jpg"
 
            cv2.imwrite(filename, rgb_image)
            print ("image saved.")
        else:
            print  (key, key == ord('s') or key == ord('S'))



    except Exception as e:
        print(f"Error: {str(e)}")

def depth_image_callback(msg):
    cv_bridge = CvBridge()
    try:
        depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        # Scale the depth image to 8-bit for visualization
        # depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)

        

        # # Compute the histogram
        # hist, bins = np.histogram(depth_image, bins=256, range=[0,256])  # Adjust the range if your depth values are different
        # # Plot the histogram
        # plt.bar(bins[:-1], hist, width=1)
        # plt.xlim([0, 256])  # Adjust this range if your depth values are different
        # plt.title('Depth Histogram')
        # plt.xlabel('Depth Value')
        # plt.ylabel('Pixel Count')
        # plt.show()


        # # Assuming the depth values are in millimeters
        # # # Define the depth range for the table
        # table_depth_min = 00  # replace with the minimum depth value for the table
        # table_depth_max = 00  # replace with the maximum depth value for the table

        # # Set depth values within the table's range to 0 (or max value, depending on your preference)
        # depth_image[(depth_image >= table_depth_min) & (depth_image <= table_depth_max)] = 0

        # hist, bins = np.histogram(depth_image, bins=256, range=[0,256])  # Adjust the range if your depth values are different
        # # Plot the histogram
        # plt.bar(bins[:-1], hist, width=1)
        # plt.xlim([0, 256])  # Adjust this range if your depth values are different
        # plt.title('Depth Histogram')
        # plt.xlabel('Depth Value')
        # plt.ylabel('Pixel Count')
        # plt.show()


        # print (depth_image.shape)
        # vision.find_box_centre_depth(depth_image,vis_output=True,vis_masks=True)

        # cv2.imshow("Depth Image", depth_image)
        # edges = cv2.Canny(depth_image, 150, 200)        
        # cv2.imshow('d_edge', edges)
        # cv2.waitKey(10)  # Adjust the delay as needed (e.g., 10 milliseconds)
    except Exception as e:
        print(f"Error: {str(e)}")




def main(args=None):
    rclpy.init(args=args)


    node = rclpy.create_node("image_visualizer")
    rgb_subscription = node.create_subscription(
        CompressedImage,
        "/myumi_005/sensors/top_azure/rgb/image_raw/compressed",  # Replace with your RGB compressed topic
        rgb_image_callback,
        1,
    )

    depth_subscription = node.create_subscription(
        Image,
        "/myumi_005/sensors/top_azure/depth/image_raw",  # Replace with your depth topic
        depth_image_callback,
        1,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



# #!/usr/bin/env python3

# import rclpy
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge
# import cv2

# def rgb_image_callback(msg):
#     cv_bridge = CvBridge()
#     try:
#         rgb_image = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
#         cv2.imshow("RGB Image", rgb_image)
#         cv2.waitKey(10)  # Adjust the delay as needed (e.g., 10 milliseconds)
#     except Exception as e:
#         print(f"Error: {str(e)}")

# def main(args=None):
#     rclpy.init(args=args)

#     node = rclpy.create_node("rgb_image_subscriber")
#     subscription = node.create_subscription(
#         CompressedImage,
#         "/myumi_005/sensors/top_azure/rgb/image_raw/compressed",  # Replace with your RGB compressed topic
#         rgb_image_callback,
#         10,
#     )

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass

#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()



# import rclpy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# def depth_image_callback(msg):
#     cv_bridge = CvBridge()
#     try:
#         depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
#         # cv2.imshow("Depth Image", depth_image)
#         # depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
#         # depth_image = depth_image.astype(np.uint8)
#         cv2.imshow("Depth Image", depth_image)
#         cv2.waitKey(1)  # Adjust the delay as needed
#         print ("a")

#     except Exception as e:
#         print(f"Error: {str(e)}")

# def main(args=None):
#     rclpy.init(args=args)

#     node = rclpy.create_node("depth_image_subscriber")
#     subscription = node.create_subscription(
#         Image,
#         "/myumi_005/sensors/top_azure/depth/image_raw",  # Replace with your depth topic
#         depth_image_callback,
#         10,
#     )

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass

#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()



# import rclpy
# from rclpy.node import Node

# from std_msgs.msg import String

# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2


# class ImageSubscriber(Node):

#     def __init__(self):
#         super().__init__('image_subscriber')
#         # self.rgb_subscription = self.create_subscription(
#         #     Image,
#         #     '/myumi_005/sensors/top_azure/rgb/image_raw/compressed',  # Replace with your actual topic
#         #     self.rgb_callback,
#         #     10)
#         # self.rgb_subscription  # Prevent unused variable warning
#         self.depth_subscription = self.create_subscription(
#             Image,
#             '/myumi_005/sensors/top_azure/depth/image_raw',  # Replace with your actual depth topic
#             self.depth_callback,
#             10)
#         self.depth_subscription  # Prevent unused variable warning
#         self.cv_bridge = CvBridge()
#         print ("The subscriber has been initialized...")


#     def rgb_callback(self, msg):
#         try:
#             cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             cv2.imshow('RGB Image', cv_image)
#             cv2.waitKey(1)  # Adjust the delay as needed
#         except Exception as e:
#             self.get_logger().error(str(e))

#     def depth_callback(self, msg):
#         try:
#             cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             cv2.imshow('RGB Image', cv_image)
#             cv2.waitKey(1)  # Adjust the delay as needed
#             # Process the depth image as needed
#         except Exception as e:
#             self.get_logger().error(str(e))


# def main(args=None):
#     rclpy.init(args=args)
#     image_subscriber = ImageSubscriber()
#     rclpy.spin(image_subscriber)
#     image_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



# # class MinimalSubscriber(Node):

# #     def __init__(self):
# #         super().__init__('minimal_subscriber')
# #         self.subscription = self.create_subscription(
# #             String,
# #             'topic',
# #             self.listener_callback,
# #             10)
# #         self.subscription  # prevent unused variable warning

# #     def listener_callback(self, msg):
# #         self.get_logger().info('I heard: "%s"' % msg.data)


# # def main(args=None):
# #     rclpy.init(args=args)

# #     minimal_subscriber = MinimalSubscriber()

# #     rclpy.spin(minimal_subscriber)

# #     # Destroy the node explicitly
# #     # (optional - otherwise it will be done automatically
# #     # when the garbage collector destroys the node object)
# #     minimal_subscriber.destroy_node()
# #     rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()