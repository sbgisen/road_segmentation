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

import math

import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class SemanticMappingNode:
    def __init__(self) -> None:
        rospy.init_node('semantic_mapping_node')

        self.camera_topic: str = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.camera_info_topic: str = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.debug: bool = rospy.get_param('~debug', False)

        self.image_sub: rospy.Subscriber = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.camera_info_sub: rospy.Subscriber = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        self.current_camera_info: CameraInfo = None
        self.occupancy_grid_pub = rospy.Publisher('~occupancy_grid', OccupancyGrid, queue_size=1)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.fov_x_range_array = None
        self.fov_y_range_array = None
        self.distance_x_map = None
        self.distance_y_map = None
        # self.grid_map = None
        self.z_difference = None
        self.x_difference = None
        self.handle_insta360_height()

        rospy.loginfo("Semantic Mapping Node Initialized")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.current_camera_info is None:
            self.current_camera_info = msg
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.calc_pixel_degree(self.current_camera_info)
            rospy.loginfo("Camera info received and FOV calculated.")
        else:
            self.current_camera_info = msg

    def image_callback(self, msg: Image) -> None:
        # rospy.loginfo("Received an image")

        if self.z_difference is None:
            rospy.logwarn("Z-axis difference is not available yet. Waiting for transform.")
            rospy.sleep(0.1)
            return

        # self.z_difference = getattr(self, 'z_difference', 1.0)

        if self.current_camera_info is None:
            rospy.logwarn("Camera info is not yet available")
            return

        if self.fov_y_range_array is None:
            rospy.logwarn("FOV y range array is not initialized.")
            self.calc_pixel_degree(self.current_camera_info)

        self.calc_pixel_distance(self.current_camera_info, self.z_difference)

        # If debug mode is enabled, publish the debug image
        if self.debug:
            self.publish_debug_image(msg)

        self.mapping_image(msg, self.current_camera_info)

    def publish_debug_image(self, debug_image: Image) -> None:
        self.debug_image_pub.publish(debug_image)

    def handle_insta360_height(self) -> None:
        # tfのリスナーを作成
        listener = tf.TransformListener()

        # ループレートを設定 (10Hz)
        rate = rospy.Rate(10.0)
        retry_count = 0
        max_retries = 10

        while not rospy.is_shutdown():
            try:
                # "base_link"フレームから"insta360_base_link"フレームへの変換を取得
                (trans, rot) = listener.lookupTransform('/base_link', '/insta360_front_optical_frame', rospy.Time(0))

                # Z軸の差分を取得
                self.z_difference = trans[2]  # trans[2]がZ軸の平行移動成分
                self.x_difference = trans[0]
                rospy.loginfo(self.x_difference)
                break

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                retry_count += 1
                rospy.logwarn("Transform not available, retrying... ({}/{})".format(retry_count, max_retries))
                if retry_count >= max_retries:
                    rospy.logerr("Transform still not available after {} retries. Please check the TF configuration."
                                 .format(max_retries))
                    break
                rospy.sleep(0.5)

            # ループを回す
            rate.sleep()

    def calc_pixel_degree(self, camera_info: CameraInfo) -> None:
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        width = camera_info.width
        height = camera_info.height

        # FOVを入れる配列をそれぞれ用意
        self.fov_x_range_array = np.zeros(width, dtype=np.float32)
        self.fov_y_range_array = np.zeros(height, dtype=np.float32)

        for x in range(width):
            norm_x = (x - cx) / fx
            fov_x = math.atan(norm_x)
            self.fov_x_range_array[x] = fov_x

        for y in range(height):
            norm_y = (y - cy) / fy
            fov_y = math.atan(norm_y)
            self.fov_y_range_array[y] = fov_y

    def calc_pixel_distance(self, camera_info: CameraInfo, z_difference: float) -> None:
        width = camera_info.width
        height = camera_info.height

        if self.fov_y_range_array is None:
            rospy.logwarn("FOV y range array is not available.")
            return

        # self.distance_map = np.zeros((height, width, 2), dtype=np.float32)
        self.distance_x_map = np.zeros((height, width), dtype=np.float32)
        self.distance_y_map = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                distance_y = z_difference / math.tan(self.fov_y_range_array[y])
                self.distance_y_map[y, x] = distance_y + self.x_difference

                distance_x = self.distance_y_map[y, x] * math.tan(self.fov_x_range_array[x])
                self.distance_x_map[y, x] = distance_x

    def mapping_image(self, msg: Image, camera_info: CameraInfo) -> None:
        # cv_bridgeを使ってImageメッセージをCV2形式の画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        height, width = cv_image.shape

        # OccupancyGridの基本情報を設定
        grid_map = OccupancyGrid()
        grid_map.header = Header()
        grid_map.header.stamp = rospy.Time.now()
        grid_map.header.frame_id = "base_link"

        # 解像度をパラメータとして取得
        resolution = rospy.get_param('~grid_resolution', 0.5)
        # 投影範囲を決定（パラメータ化）
        ground_width = rospy.get_param('~ground_width', 40)  # [m]
        ground_height = rospy.get_param('~ground_height', 40)  # [m]

        # OccupancyGridの幅と高さを設定
        grid_map.info.resolution = resolution
        grid_width = int(ground_width / resolution)
        grid_height = int(ground_height / resolution)
        grid_map.info.width = grid_width
        grid_map.info.height = grid_height

        # グリッドの原点を設定（カメラ画像を地面に投影したとき中央手前に設定）
        # origin_x = self.distance_y_map[-1, mid_width_index]
        origin_tfx = 0
        origin_tfy = - ground_height / 2
        grid_map.info.origin = Pose(Point(origin_tfx, origin_tfy, 0), Quaternion(0, 0, 0, 1))
        grid_data = np.full((grid_width, grid_height), -1, dtype=np.int8)

        start_y = int(camera_info.height / 2 + camera_info.height * 0.03)
        # 中心よりy軸方向のsizeの3%ほど下の部分を投影する
        # for y in range(start_y, height, 1):
        for x in range(width):
            for y in range(start_y, height, 1):
                # for x in range(width):
                pixel_value = cv_image[y, x]
                if pixel_value == 255:
                    pixel_value = 0
                else:
                    pixel_value = 100
                grid_x = int((self.distance_y_map[y, x]) / resolution)
                grid_y = (grid_height - int((self.distance_x_map[y, x] - origin_tfy) / resolution) - 1)
                if 0 <= grid_x < grid_height and 0 <= grid_y < grid_width:
                    grid_data[grid_y, grid_x] = pixel_value

            # rospy.loginfo("%d, %d, %f", x, y, self.distance_y_map[y, x])
        grid_map.data = grid_data.flatten().tolist()
        self.occupancy_grid_pub.publish(grid_map)


if __name__ == '__main__':
    try:
        node = SemanticMappingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
