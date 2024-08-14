#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np
import rospy
import tensorflow as tensorflow
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from road_segmentation.data_loader.display import create_mask

bridge = CvBridge()


class RoadSegmentationNode:
    def __init__(self):
        rospy.init_node('road_segmentation_node')

        # GPU configuration
        self.gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        if self.gpus:
            try:
                tensorflow.config.experimental.set_virtual_device_configuration(
                    self.gpus[0],
                    [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=500)]
                )
            except RuntimeError as e:
                rospy.logerr("Error configuring GPU: {}".format(e))

        # Set default model path to pspunet_weight.h5 in the models directory
        default_model_path = os.path.join(
            rospy.get_param('~model_dir', os.path.dirname(__file__) + '/../models'),
            'pspunet_weight.h5'
        )
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.model_path = rospy.get_param('~model_path', default_model_path)
        self.debug = rospy.get_param('~debug', True)

        # Load class names and labels from ROS parameter
        class_data = rospy.get_param('class')
        self.class_names = list(class_data.keys())
        self.num_classes = len(self.class_names)

        # Load the segmentation model
        self.model = tensorflow.keras.models.load_model(self.model_path)

        # Get input image size from model
        self.input_image_size = self.model.input_shape[1:3]  # (height, width)

        # Subscribe to the camera and camera_info topics
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)

        # Publisher for adjusted camera info and the segmentation result for each class
        self.camera_info_pub = rospy.Publisher('~result/camera_info', CameraInfo, queue_size=1)
        # Publisher for the segmentation result for each class
        self.result_pubs = []
        for class_name in self.class_names:
            pub = rospy.Publisher(f'~result/{class_name}_mask', Image, queue_size=1)
            self.result_pubs.append(pub)

        # Publisher for debug image (only if debug mode is enabled)
        if self.debug:
            self.debug_image_pub = rospy.Publisher('~debug_image', Image, queue_size=1)

        self.current_camera_info = None
        rospy.loginfo("Road Segmentation Node Initialized")

    def camera_info_callback(self, msg):
        self.current_camera_info = msg

    def perform_segmentation(self, image):
        # Resize the input image to the size expected by the model
        input_image = cv2.resize(image, (self.input_image_size[1], self.input_image_size[0]))  # (width, height)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image / 255.0  # Normalize input

        # Get the segmentation result (class map)
        result = self.model.predict(input_image)

        # Convert the result to a mask image using create_mask
        result_mask = create_mask(result).numpy()

        return result_mask

    def create_debug_image(self, image, result_mask):
        # Apply colormap to visualize the mask
        result_mask_colored = cv2.applyColorMap((result_mask * 36).astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay the segmentation result on the original image with transparency
        alpha = 0.6
        debug_image = cv2.addWeighted(image, alpha, result_mask_colored, 1 - alpha, 0)

        return debug_image

    def publish_debug_image(self, debug_image):
        try:
            # Convert the CV2 image back to a ROS Image message
            debug_image_msg = bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_image_pub.publish(debug_image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def adjust_camera_info(self, original_info, target_width, target_height):
        scale_x = target_width / float(original_info.width)
        scale_y = target_height / float(original_info.height)

        adjusted_info = CameraInfo()
        adjusted_info.header = original_info.header
        adjusted_info.width = target_width
        adjusted_info.height = target_height
        adjusted_info.K = [scale_x * original_info.K[0], 0, scale_x * original_info.K[2],
                           0, scale_y * original_info.K[4], scale_y * original_info.K[5],
                           0, 0, 1]
        adjusted_info.P = [scale_x * original_info.P[0], 0, scale_x * original_info.P[2], 0,
                           0, scale_y * original_info.P[5], scale_y * original_info.P[6], 0,
                           0, 0, 1, 0]
        adjusted_info.D = original_info.D
        adjusted_info.R = original_info.R

        return adjusted_info

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Perform segmentation on the received image
        result_mask = self.perform_segmentation(cv_image)

        # Resize the entire mask to match the original image size
        result_mask_resized = cv2.resize(
            result_mask,
            (cv_image.shape[1],
             cv_image.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        # Adjust the camera info for the resized mask
        if self.current_camera_info is not None:
            adjusted_camera_info = self.adjust_camera_info(
                self.current_camera_info, cv_image.shape[1], cv_image.shape[0])
            self.camera_info_pub.publish(adjusted_camera_info)

        # Publish the mask for each class
        for i in range(self.num_classes):
            class_mask = (result_mask_resized == i).astype(np.uint8) * 255
            self.publish_class_mask(class_mask, self.result_pubs[i])

        # If debug mode is enabled, publish the debug image
        if self.debug:
            debug_image = self.create_debug_image(cv_image, result_mask_resized)
            self.publish_debug_image(debug_image)

    def publish_class_mask(self, class_mask, pub):
        try:
            # Convert the CV2 image back to a ROS Image message
            mask_msg = bridge.cv2_to_imgmsg(class_mask, encoding='mono8')
            pub.publish(mask_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


if __name__ == '__main__':
    try:
        node = RoadSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
