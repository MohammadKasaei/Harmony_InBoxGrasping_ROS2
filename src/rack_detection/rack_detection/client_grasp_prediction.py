import sys

# from example_interfaces.srv import AddTwoInts
from rack_detection_interface.srv import RackDetection

import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(RackDetection, 'grasp_pose_detection')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RackDetection.Request()


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

    

    response = minimal_client.send_request(dbg_vis, show_masks_point,show_raw_image )
    minimal_client.get_logger().info(
        'Params : %d , %d , %d response: pos:(%f,%f,%f), angle: %f, prob: %f, exec: %d' %
        (dbg_vis, show_masks_point,show_raw_image , \
         response.pose.position.x,response.pose.position.y,response.pose.position.z,
         response.pose.orientation.z,
         response.probablity,
         response.execute))

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()