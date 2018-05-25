YOLO_quantize is designed for embedded device to run YOLOv1(by the father of YOLO, Joseph Redmon) smoothly. for now, it is verified on ZCU102 SoC device. 
The repo is still under development. Here is developing features:
1. support build options for x86 and ARM devices [Done]
   e.g. Options are ARM and X86 in Makefile and fit CC and CPP fir your local path.
2. support YOLO inference on Webcam YUV422 format. [Done]
   e.g. you could run it on PC with the following cmd.
   ./darknet zcu102 resource/yuyv422_640480_car.yuv 640x480
3. apply NEON acceleration for convolution layer, using EIGEN library [Done]
   e.g. Enable EIGEN option in Makefile.
4. support simple layer profiling[Done]
   e.g. PROF config in Makefile
5. separate YOLO init and inference flow.[Done]
   e.g. see validate_zcu102() for reference.
6. support 16-bit trainning and inference on convolution layer [on-going]
   e.g. fixing segamentation fault
7. support PCA for trainning and inference [planning]
8. acceleration connected layer and img2col() [planning]

You can do what the fuck you want, but limited for research. 
If it helps you, please cite it!  :)
