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
        self.configure_gpu()

        # Parameters
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.model_path = rospy.get_param('~model_path', self.get_default_model_path())
        self.debug = rospy.get_param('~debug', True)

        # Load model and set class information
        self.model = tensorflow.keras.models.load_model(self.model_path)
        self.class_names = self.load_class_names()
        self.num_classes = len(self.class_names)
        self.trained_image_size = self.model.input_shape[1:3]  # (height, width)

        # Set up ROS topics
        self.setup_ros_topics()

        rospy.loginfo("Road Segmentation Node Initialized")

    def configure_gpu(self):
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tensorflow.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=500)]
                )
            except RuntimeError as e:
                rospy.logerr(f"Error configuring GPU: {e}")

    def get_default_model_path(self):
        return os.path.join(
            rospy.get_param(
                '~model_dir',
                os.path.dirname(__file__)
                + '/../models'),
            'pspunet_weight.h5')

    def load_class_names(self):
        class_data = rospy.get_param('class')
        return list(class_data.keys())

    def setup_ros_topics(self):
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.camera_info_pub = rospy.Publisher('~result/camera_info', CameraInfo, queue_size=1)
        self.result_pubs = [rospy.Publisher(f'~result/{name}_mask', Image, queue_size=1)
                            for name in self.class_names]
        if self.debug:
            self.debug_image_pub = rospy.Publisher('~debug_image', Image, queue_size=1)

    def camera_info_callback(self, msg):
        self.current_camera_info = msg

    def crop_and_resize_image(self, original_image, target_size):
        height, width, _ = original_image.shape
        target_height, target_width = target_size

        input_aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height

        if input_aspect_ratio > target_aspect_ratio:
            new_width = int(target_aspect_ratio * height)
            offset = (width - new_width) // 2
            cropped_image = original_image[:, offset:offset + new_width]
        else:
            new_height = int(width / target_aspect_ratio)
            offset = (height - new_height) // 2
            cropped_image = original_image[offset:offset + new_height, :]

        resized_image = cv2.resize(cropped_image, (target_width, target_height))
        return resized_image

    def crop_and_resize_camera_info(self, original_camera_info, target_size):
        original_width = original_camera_info.width
        original_height = original_camera_info.height

        target_width, target_height = target_size
        target_aspect_ratio = target_width / target_height
        input_aspect_ratio = original_width / original_height

        if input_aspect_ratio > target_aspect_ratio:
            new_width = int(target_aspect_ratio * original_height)
            crop_offset_x = (original_width - new_width) // 2
            crop_offset_y = 0
            cropped_width = new_width
            cropped_height = original_height
        else:
            new_height = int(original_width / target_aspect_ratio)
            crop_offset_x = 0
            crop_offset_y = (original_height - new_height) // 2
            cropped_width = original_width
            cropped_height = new_height

        scale_x = target_width / float(cropped_width)
        scale_y = target_height / float(cropped_height)

        adjusted_info = CameraInfo()
        adjusted_info.header = original_camera_info.header
        adjusted_info.width = target_width
        adjusted_info.height = target_height

        adjusted_info.K = [
            scale_x * original_camera_info.K[0], 0, scale_x * (original_camera_info.K[2] - crop_offset_x),
            0, scale_y * original_camera_info.K[4], scale_y * (original_camera_info.K[5] - crop_offset_y),
            0, 0, 1
        ]

        adjusted_info.P = [
            scale_x * original_camera_info.P[0], 0, scale_x * (original_camera_info.P[2] - crop_offset_x), 0,
            0, scale_y * original_camera_info.P[5], scale_y * (original_camera_info.P[6] - crop_offset_y), 0,
            0, 0, 1, 0
        ]

        adjusted_info.D = original_camera_info.D
        adjusted_info.R = original_camera_info.R

        return adjusted_info

    def perform_segmentation(self, image):
        input_image = self.crop_and_resize_image(image, self.trained_image_size)
        input_image = input_image[tensorflow.newaxis, ...] / 255.0
        result = self.model.predict(input_image)
        return create_mask(result).numpy()

    def create_debug_image(self, image, result_mask):
        overlay_image = image / 2
        overlay_image[(result_mask == 0)] += [0, 0, 0]  # Class 0: "Background"
        overlay_image[(result_mask == 1)] += [20, 20, 20]  # Class 1: "Bike_lane"
        overlay_image[(result_mask == 2)] += [100, 100, 0]  # Class 2: "Caution_zone"
        overlay_image[(result_mask == 3)] += [40, 140, 100]  # Class 3: "Crosswalk"
        overlay_image[(result_mask == 4)] += [0, 100, 100]  # Class 4: "braille_guide_blocks"
        overlay_image[(result_mask == 5)] += [0, 0, 100]  # Class 5: "Roadway"
        overlay_image[(result_mask == 6)] += [100, 0, 0]  # Class 6: "Sidewalk"
        return np.uint8(np.clip(overlay_image, 0, 255))

    def publish_debug_image(self, debug_image):
        try:
            debug_image_msg = bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_image_pub.publish(debug_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def publish_class_mask(self, class_mask, pub):
        try:
            mask_msg = bridge.cv2_to_imgmsg(class_mask, encoding='mono8')
            pub.publish(mask_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def image_callback(self, msg):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        resized_image = self.crop_and_resize_image(cv_image, self.trained_image_size)
        result_mask = self.perform_segmentation(resized_image)
        result_mask_resized = cv2.resize(
            result_mask,
            (resized_image.shape[1],
             resized_image.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        if self.current_camera_info is not None:
            adjusted_camera_info = self.crop_and_resize_camera_info(
                self.current_camera_info, (resized_image.shape[1], resized_image.shape[0]))
            self.camera_info_pub.publish(adjusted_camera_info)

        for i in range(self.num_classes):
            class_mask = (result_mask_resized == i).astype(np.uint8) * 255
            self.publish_class_mask(class_mask, self.result_pubs[i])

        if self.debug:
            debug_image = self.create_debug_image(resized_image, result_mask_resized)
            self.publish_debug_image(debug_image)


if __name__ == '__main__':
    try:
        node = RoadSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
