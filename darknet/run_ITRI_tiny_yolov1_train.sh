#! /bin/sh
#
# run_ITRI_tiny_yolov1.sh
# Copyright (C) 2018 lucas <lucas@lucas>
#
# Distributed under terms of the MIT license.
#

./darknet yolo train cfg/yolov1/yolo.train.cfg extraction.conv.weights
