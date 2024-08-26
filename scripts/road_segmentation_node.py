#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np
import rospy
import tensorflow as tensorflow
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from road_segmentation.data_loader.display import create_mask
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

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
        self.trained_image_size = self.model.input_shape[1:3]  # (height, width)

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

    def crop_and_resize_image(self, image, target_size):
        # Get the dimensions of the input image
        height, width, _ = image.shape
        target_height, target_width = target_size

        # Calculate the aspect ratios
        input_aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height

        # Crop the image to match the aspect ratio of the target size
        if input_aspect_ratio > target_aspect_ratio:
            # Input image is wider than the target aspect ratio
            new_width = int(target_aspect_ratio * height)
            offset = (width - new_width) // 2
            cropped_image = image[:, offset:offset + new_width]
        else:
            # Input image is taller than the target aspect ratio
            new_height = int(width / target_aspect_ratio)
            offset = (height - new_height) // 2
            cropped_image = image[offset:offset + new_height, :]

        # Resize the cropped image to the target size
        resized_image = cv2.resize(cropped_image, (target_width, target_height))  # (width, height)

        return resized_image

    def perform_segmentation(self, image):
        # Crop and resize the input image to the size expected by the model
        input_image = self.crop_and_resize_image(image, self.trained_image_size)

        input_image = input_image[tensorflow.newaxis, ...]
        input_image = input_image / 255.0  # Normalize input

        # Get the segmentation result (class map)
        result = self.model.predict(input_image)

        # Convert the result to a mask image using create_mask
        result_mask = create_mask(result).numpy()

        return result_mask

    def create_debug_image(self, image, result_mask):
        # Create a copy of the original image and reduce its brightness by half
        overlay_image = image / 2

        # Apply specific colors to each class
        overlay_image[(result_mask == 0)] += [0, 0, 0]  # Class 0: "Background"
        overlay_image[(result_mask == 1)] += [50, 0, 50]  # Class 1: "Bike_lane"
        overlay_image[(result_mask == 2)] += [100, 100, 0]  # Class 2: "Caution_zone"
        overlay_image[(result_mask == 3)] += [40, 140, 100]  # Class 3: "Crosswalk"
        overlay_image[(result_mask == 4)] += [0, 100, 100]  # Class 4: "braille_guide_blocks"
        overlay_image[(result_mask == 5)] += [0, 0, 100]  # Class 5: "Roadway"
        overlay_image[(result_mask == 6)] += [100, 0, 0]  # Class 6: "Sidewalk"

        # Ensure that the values are within the valid range [0, 255]
        overlay_image = np.clip(overlay_image, 0, 255)

        # Convert the image to uint8
        debug_image = np.uint8(overlay_image)

        return debug_image

    def publish_debug_image(self, debug_image):
        try:
            # Convert the CV2 image back to a ROS Image message
            debug_image_msg = bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_image_pub.publish(debug_image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def crop_and_resize_camera_info(self, original_info, target_width, target_height):
        # Get original image size from CameraInfo
        original_width = original_info.width
        original_height = original_info.height

        # Calculate aspect ratios
        target_aspect_ratio = target_width / target_height
        input_aspect_ratio = original_width / original_height

        # Determine crop offset and size based on aspect ratio
        if input_aspect_ratio > target_aspect_ratio:
            # If the image is wider than the target aspect ratio, crop width
            new_width = int(target_aspect_ratio * original_height)
            crop_offset_x = (original_width - new_width) // 2
            crop_offset_y = 0
            cropped_width = new_width
            cropped_height = original_height
        else:
            # If the image is taller than the target aspect ratio, crop height
            new_height = int(original_width / target_aspect_ratio)
            crop_offset_x = 0
            crop_offset_y = (original_height - new_height) // 2
            cropped_width = original_width
            cropped_height = new_height

        # Calculate scaling factors for resizing
        scale_x = target_width / float(cropped_width)
        scale_y = target_height / float(cropped_height)

        # Adjust the camera information
        adjusted_info = CameraInfo()
        adjusted_info.header = original_info.header
        adjusted_info.width = target_width
        adjusted_info.height = target_height

        # Adjust the K matrix (intrinsic camera matrix)
        adjusted_info.K = [
            scale_x * original_info.K[0], 0, scale_x * (original_info.K[2] - crop_offset_x),
            0, scale_y * original_info.K[4], scale_y * (original_info.K[5] - crop_offset_y),
            0, 0, 1
        ]

        # Adjust the P matrix (projection matrix)
        adjusted_info.P = [
            scale_x * original_info.P[0], 0, scale_x * (original_info.P[2] - crop_offset_x), 0,
            0, scale_y * original_info.P[5], scale_y * (original_info.P[6] - crop_offset_y), 0,
            0, 0, 1, 0
        ]

        # Keep other camera parameters unchanged
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

        resized_image = self.crop_and_resize_image(cv_image, self.trained_image_size)

        # Perform segmentation on the resized image
        result_mask = self.perform_segmentation(resized_image)

        # Resize the entire mask to match the original resized image size
        result_mask_resized = cv2.resize(
            result_mask,
            (resized_image.shape[1], resized_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Adjust the camera info based on the cropped and resized image
        if self.current_camera_info is not None:
            adjusted_camera_info = self.crop_and_resize_camera_info(
                self.current_camera_info, resized_image.shape[1], resized_image.shape[0]
            )
            self.camera_info_pub.publish(adjusted_camera_info)

        # Publish the mask for each class
        for i in range(self.num_classes):
            class_mask = (result_mask_resized == i).astype(np.uint8) * 255
            self.publish_class_mask(class_mask, self.result_pubs[i])

        # If debug mode is enabled, publish the debug image using the resized image
        if self.debug:
            debug_image = self.create_debug_image(resized_image, result_mask_resized)
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
