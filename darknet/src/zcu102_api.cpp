#include "darknet.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yolo_lib.h>

using namespace cv;

static image **alphabet;

static const char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


static network *net;
void setup_yolo_env(char *cfgfile, char *weightfile)
{
	printf("setup yolo_env\n");
	net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
	alphabet = load_alphabet();
}

void set_cur_img_by_name(char *filename)
{
	char *p;
	net->cur_im = load_image_color(filename,0,0);

	strncpy(net->cur_im.name, filename, 256);
	return;
}

static float *image_normalization(void *p, int w, int h, int c)
{
	int i,j,k;
	char *char_data = p;

	float *data = malloc(sizeof(float)*w*h*c);
	for(k = 0; k < c; ++k){
		for(j = 0; j < h; ++j){
			for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                data[dst_index] = (float)char_data[src_index]/255.;
            }    
        }    
    }

	return data;
}

void set_cur_img(unsigned char *data, int w, int h, int c, const char *name)
{
	net->cur_im.data= (float *)data;
	net->cur_im.w = w;
	net->cur_im.h = h;
	strncpy((net->cur_im.name), name, 256);
	return;
}


float *yolo_inference(float thresh)
{

    clock_t time;
    layer l = net->layers[net->n-1];

	image sized = resize_image(net->cur_im, net->w, net->h);
	float *X = sized.data;
	time=clock();
	network_predict(net, X);
	printf("%s: Predicted in %f seconds.\n", net->cur_im.name, sec(clock()-time));
	return l.output;
}


void write_jpeg(Mat &a, const char *filename)
{
	std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

	imwrite(filename, a, compression_params);
}

void write_raw_image(void *raw_image,const char *filename, unsigned int size)
{
	FILE *out_fp;
	out_fp = fopen(filename, "wb");
	fwrite(raw_image,1,size,out_fp);
	fclose(out_fp);

}


void validate_zcu102(int argc, char **argv)
{
    int width;
    int height;
	FILE *fp;
	void *res = argv[3];
	char *file = argv[2];
	long f_size;	
	void *yuv422_data;
	size_t ret;

    if(argc < 6){
        fprintf(stderr, "usage: ./darknet zcu102 [input] [input resolution]\n");
        fprintf(stderr, "ex: ./darknet zcu102 resource/yuyv422_640480_car.yuv  640x480\n");
        return 0;
    }



    width = atoi(strtok(res, "x"));
    height = atoi(strtok(NULL, "x"));
	printf("%s, %dx%d\n", file, width,height);

   /*
     * Set up YUV422 input image.
     * */
    fp = fopen(file, "rb");
    if (fp==NULL) {
        fputs ("File error",stderr);
        exit (1);
    }

    fseek (fp, 0, SEEK_END);   // non-portable
    f_size=ftell(fp);
    rewind(fp);

    yuv422_data = malloc(f_size);
    ret = fread(yuv422_data, 1, f_size, fp);

    printf("read %ld bytes, error %d\n", ret, ferror (fp));
    fclose(fp);

    setup_yolo_env(argv[4], argv[5]);
//	setup_yolo_env("resource/R-tiny_v1.cfg", "resource/R-tiny_v1.weights");
    yolo_inference_with_ptr(yuv422_data, width, height, 2, 0.2);

}


int yolo_inference_with_ptr(void *ptr, int w, int h, int c, float thresh)
{

    void *yuv422_data;
    void *rgb24_data;
    void *resiz_data;
	clock_t time;
	clock_t overall;

	overall = clock();

	layer l = net->layers[net->n-1];
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(int j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));


	yuv422_data = ptr;
	Mat YUV422_Mat(h, w, CV_8UC2, yuv422_data);

	rgb24_data = malloc(h*w*3);
	Mat RGB24_Mat(h ,w ,CV_8UC3,rgb24_data);
	cvtColor(YUV422_Mat, RGB24_Mat, COLOR_YUV2RGB_YUYV);

	resiz_data = malloc(net->h*net->w*net->c);
	Mat Resize_Mat(net->h, net->w,CV_8UC3, resiz_data);
	resize(RGB24_Mat, Resize_Mat, Size(net->h, net->w));

	printf("%d,%d,%d\n", net->h, net->w, net->c);

#if DEBUG
	write_raw_image(yuv422_data,"yuv422_640480_car_conv.raw",w*h*2 );
	write_raw_image(rgb24_data,"rgb24_640480_car_conv.raw",w*h*3 );
	write_raw_image(resiz_data,"rgb24_448x448_car_conv.raw", net->h*net->w*net->c );

	/*
	 * Resize_Mat.ptr() = resiz_data.
	 * */
	write_raw_image(Resize_Mat.ptr(),"mptr_rgb24_448x448_car_conv.raw",net->h*net->w*net->c );
	/*
	 * imwrite() only accepts BGR format.
	 * */
	cvtColor(RGB24_Mat, RGB24_Mat, COLOR_RGB2BGR);
	write_jpeg(RGB24_Mat, "rgb24_640480_car_conv.jpg");
	cvtColor(Resize_Mat, Resize_Mat, COLOR_RGB2BGR);
	write_jpeg(Resize_Mat, "rgb24_448x448_car_conv.jpg");
#endif



	image input;
	input.w = net->w;
	input.h = net->h;
	input.c = net->c;
	input.data = image_normalization(Resize_Mat.ptr(), input.w, input.h, input.c);
	time=clock();
	network_predict(net, input.data);
	printf("Predicted in %f seconds.\n", sec(clock()-time));

	get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0); 
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms); /*eliminate the similar box and only left 1 box*/

	image unscale_input;
	unscale_input.w = w ; 
	unscale_input.h = h ;
	unscale_input.c = 3 ;
	unscale_input.data = image_normalization(RGB24_Mat.ptr(), unscale_input.w, unscale_input.h, unscale_input.c);

	draw_detections(unscale_input, l.side*l.side*l.n, thresh, boxes, probs, 0, voc_names, alphabet, 20);
	save_image(unscale_input, "predictions");
	printf("overall time in %f seconds.\n", sec(clock()-overall));

	free_image(input);

}


