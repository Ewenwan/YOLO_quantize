#! /bin/sh
#
# run_ITRI_tiny_yolov1.sh
# Copyright (C) 2018 lucas <lucas@lucas>
#
# Distributed under terms of the MIT license.
#


./darknet yolo test cfg/yolov1/yolo-tiny_v1.cfg yolo-tiny_v1.weights data/dog.jpg
