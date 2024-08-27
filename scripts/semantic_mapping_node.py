#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2024 SoftBank Corp.
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

# 目標：画像データを取得し、Rvizで表示できるようにする。

# import cv2
import rospy
from cv_bridge import CvBridge
# from cv_bridge import CvBridgeError
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

bridge = CvBridge()


class SemanticMappingNode:
    def __init__(self) -> None:
        rospy.init_node('semantic_mapping_node')

        self.camera_topic: str = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.camera_info_topic: str = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.debug: bool = rospy.get_param('~debug', True)

        self.image_sub: rospy.Subscriber = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.camera_info_sub: rospy.Subscriber = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        if self.debug:
            self.debug_image_pub: rospy.Publisher = rospy.Publisher('~debug_image', Image, queue_size=1)

        self.current_camera_info: CameraInfo | None = None
        rospy.loginfo("Semantic Mapping Node Initialized")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.current_camera_info = msg

    def image_callback(self, msg: Image) -> None:
        rospy.loginfo("Received an image")
        # If debug mode is enabled, publish the debug image
        if self.debug:
            debug_image: Image = msg
            self.publish_debug_image(debug_image)

    def publish_debug_image(self, debug_image: Image) -> None:
        self.debug_image_pub.publish(debug_image)


if __name__ == '__main__':
    try:
        node = SemanticMappingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
