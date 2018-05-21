#! /bin/sh
#
# run_ITRI_tiny_yolov1.sh
# Copyright (C) 2018 lucas <lucas@lucas>
#
# Distributed under terms of the MIT license.
#


#./darknet zcu102 resource/yuyv422_640480_car.yuv 640x480 resource/R-tiny_v1.cfg resource/R-tiny_v1.weights
./darknet zcu102 resource/yuyv422_640480_car.yuv 640x480 resource/tiny-yolo-xnor.cfg resource/tiny-yolo-xnor.weight
