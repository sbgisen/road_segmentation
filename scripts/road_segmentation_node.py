#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np
import rospy
import tensorflow as tensorflow
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from jsk_recognition_msgs.msg import ClassificationResult
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
        self.model_path = rospy.get_param('~model_path', default_model_path)
        self.debug = rospy.get_param('~debug', False)

        # Load the segmentation model
        self.model = tensorflow.keras.models.load_model(self.model_path)

        # Get input image size from model
        self.input_image_size = self.model.input_shape[1:3]  # (height, width)

        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)

        # Publisher for the segmentation result
        self.result_pub = rospy.Publisher('/road_segmentation/result', ClassificationResult, queue_size=1)

        # Publisher for debug image (only if debug mode is enabled)
        if self.debug:
            self.debug_image_pub = rospy.Publisher('/road_segmentation/debug_image', Image, queue_size=1)

        rospy.loginfo("Road Segmentation Node Initialized")

    def perform_segmentation(self, image):
        # Resize the input image to the size expected by the model
        input_image = cv2.resize(image, (self.input_image_size[1], self.input_image_size[0]))  # (width, height)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image / 255.0  # Normalize input

        # Get the segmentation result (class map)
        result = self.model.predict(input_image)

        # Convert the result to a mask image using create_mask
        result_mask = create_mask(result).numpy()

        return result, result_mask

    def create_debug_image(self, image, result_mask):
        # Resize result_mask to match the original image size
        result_mask = cv2.resize(result_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

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

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Perform segmentation on the received image
        result, result_mask = self.perform_segmentation(cv_image)

        # Publish result only if it has meaningful content
        if np.any(result_mask):
            self.publish_result(result)

        # If debug mode is enabled, publish the debug image
        if self.debug:
            debug_image = self.create_debug_image(cv_image, result_mask)
            self.publish_debug_image(debug_image)

    def publish_result(self, result):
        # Create and fill ClassificationResult message
        classification_result = ClassificationResult()
        # Fill classification_result with actual data here if necessary
        # Example: classification_result.labels = [list of labels]

        # Only publish if there is content
        if classification_result.labels:
            self.result_pub.publish(classification_result)


if __name__ == '__main__':
    try:
        node = RoadSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
