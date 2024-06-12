import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__("stereocamera_node")
        self.left_image_topic_ = self.declare_parameter("left_image_topic", "/image/left/image_compressed").value
        self.right_image_topic_ = self.declare_parameter("right_image_topic", "/image/right/image_compressed").value
        self.exception_counter = 0
        self.fps = 5  # Adjust FPS as necessary
        self.left_frame_id_ = self.declare_parameter("left_frame_id", "left_camera").value
        self.right_frame_id_ = self.declare_parameter("right_frame_id", "right_camera").value
        self.left_image_publisher_ = self.create_publisher(CompressedImage, self.left_image_topic_, 1)
        self.right_image_publisher_ = self.create_publisher(CompressedImage, self.right_image_topic_, 1)
        
        self.br = CvBridge()

        self.timer = self.create_timer(1.0/self.fps, self.image_callback)
        self.get_logger().info(f"Stereocamera node ready with {self.fps} FPS")


    def image_callback(self):
        factor=1
        try:
            time_msg = self.get_time_msg()
            if hasattr(self, 'first_plt_img') and hasattr(self, 'sec_plt_img'):

                # Resize images based on the factor
                resized_first_img = cv2.resize(self.first_plt_img, 
                                               (int(self.first_plt_img.shape[1] * factor), 
                                                int(self.first_plt_img.shape[0] * factor)))
                resized_sec_img = cv2.resize(self.sec_plt_img, 
                                             (int(self.sec_plt_img.shape[1] * factor), 
                                              int(self.sec_plt_img.shape[0] * factor)))

                # Create black frames
                frame_height = max(self.first_plt_img.shape[0], self.sec_plt_img.shape[0]) * 2
                frame_width = max(self.first_plt_img.shape[1], self.sec_plt_img.shape[1]) * 2
                black_frame_first = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                black_frame_sec = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                # Position resized images in the center of black frames
                resized_first_y = ((frame_height - resized_first_img.shape[0]) // 2) - 20
                resized_first_x = (frame_width - resized_first_img.shape[1]) // 2
                black_frame_first[resized_first_y:resized_first_y + resized_first_img.shape[0], 
                                  resized_first_x:resized_first_x + resized_first_img.shape[1]] = resized_first_img

                resized_sec_y = (frame_height - resized_sec_img.shape[0]) // 2
                resized_sec_x = (frame_width - resized_sec_img.shape[1]) // 2
                black_frame_sec[resized_sec_y:resized_sec_y + resized_sec_img.shape[0], 
                                resized_sec_x:resized_sec_x + resized_sec_img.shape[1]] = resized_sec_img

                # Convert frames to image messages
                left_img_msg = self.get_image_msg(black_frame_first, time_msg, compressed=True)
                right_img_msg = self.get_image_msg(black_frame_sec, time_msg, compressed=True, left=False)

                # Publish the image messages
                self.left_image_publisher_.publish(left_img_msg)
                self.right_image_publisher_.publish(right_img_msg)


        except Exception as e:
            # Handle exceptions if needed
            print(f"Error in image_callback: {e}")

    def get_time_msg(self):
        time_msg = self.get_clock().now().to_msg()
        return time_msg

    def get_image_msg(self, image, time, compressed=False, left=True):
        img_msg = self.br.cv2_to_compressed_imgmsg(image, dst_format='jpeg') if compressed else self.br.cv2_to_imgmsg(image)
        img_msg.header.stamp = time
        img_msg.header.frame_id = self.left_frame_id_ if left else self.right_frame_id_
        return img_msg

def main(args=None):
    rclpy.init(args=args)
    publisher_node = CameraNode.get_instance()
    rclpy.spin(publisher_node)
    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
