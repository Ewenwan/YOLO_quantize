#! /bin/sh
#
# run_ITRI_tiny_yolov1.sh
# Copyright (C) 2018 lucas <lucas@lucas>
#
# Distributed under terms of the MIT license.
#


./darknet yolo test resource/R-tiny_v1.cfg resource/R-tiny_v1.weights data/dog.jpg
