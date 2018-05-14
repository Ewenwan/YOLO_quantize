#ifndef IM2COL_H
#define IM2COL_H
#include "darknet.h"


void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

void im2col_cpu_quant(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, quant_t* data_col);
#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#endif
