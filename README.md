YOLO_quantize is designed for embedded device to run YOLOv1(by the father of YOLO, Joseph Redmon) smoothly. 
for now, it is verified on 赛灵思（Xilinx） ZCU102 SoC device. 
The repo is still under development. 
Here is developing features:
1. 编译支持 support build options for x86 and ARM devices 
   e.g. Options are ARM and X86 in Makefile and make CC and CPP to fit your local path.
2. 视频数据格式 support YOLO inference on Webcam YUV422 format. 
   e.g. you could run it on PC with the following cmd.
   ./darknet zcu102 resource/yuyv422_640480_car.yuv 640x480
3. ARM NENO加速 apply NEON acceleration for convolution layer, using EIGEN library 
   e.g. Enable EIGEN option in Makefile.
4. support simple layer profiling
   e.g. PROF config in Makefile
5. separate YOLO init and inference flow.
   e.g. see validate_zcu102() for reference.
6. 16位量化 support 16-bit trainning and inference on convolution layer [on-going]
   e.g. fixing segamentation fault
7. PCA压缩 support PCA for trainning and inference [planning]
8. 卷积矩阵乘法加速 acceleration connected layer and img2col() [planning]

You can do anything you want, but limited for research. 
If it helps you, please cite it!  :)



[电子创新网 赛灵思中文社区](http://xilinx.eetrend.com/)

