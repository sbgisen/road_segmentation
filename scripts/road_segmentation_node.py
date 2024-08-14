#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from road_segmentation.model_loader import predict
from road_segmentation.model_loader import load_model
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import RectArray
from jsk_recognition_msgs.msg import ClassificationResult
from cv_bridge import CvBridgeError
from cv_bridge import CvBridge
import tensorflow as tensorflow
import rospy
import numpy as np
import cv2
import os

print("Python interpreter:", os.popen('which python3').read())


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

        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)

        # Publisher for the segmentation result
        self.result_pub = rospy.Publisher('/road_segmentation/result', ClassificationResult, queue_size=1)

        # Publisher for debug image (only if debug mode is enabled)
        if self.debug:
            self.debug_image_pub = rospy.Publisher('/road_segmentation/debug_image', Image, queue_size=1)

        # Load the segmentation model
        self.model = load_model(self.model_path)

        rospy.loginfo("Road Segmentation Node Initialized")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Perform segmentation on the received image
        prediction = self.perform_segmentation(cv_image)
        self.publish_result(prediction)

        # If debug mode is enabled, publish the debug image
        if self.debug:
            debug_image = self.create_debug_image(cv_image, prediction)
            self.publish_debug_image(debug_image)

    def perform_segmentation(self, image):
        # Resize the input image and perform prediction using the model
        input_image = cv2.resize(image, (256, 256))
        input_image = np.expand_dims(input_image, axis=0)
        result = predict(self.model, input_image)
        result = cv2.resize(result[0], (image.shape[1], image.shape[0]))
        return result

    def publish_result(self, result):
        # Publish the segmentation result
        classification_result = ClassificationResult()
        # Fill classification_result with actual data here

        self.result_pub.publish(classification_result)

    def create_debug_image(self, image, result):
        # Create a debug image overlaying the result on the original image
        debug_image = cv2.addWeighted(image, 0.7, result, 0.3, 0)
        return debug_image

    def publish_debug_image(self, debug_image):
        try:
            # Convert the CV2 image back to a ROS Image message
            debug_image_msg = bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_image_pub.publish(debug_image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


if __name__ == '__main__':
    try:
        node = RoadSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
