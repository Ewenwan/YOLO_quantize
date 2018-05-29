#include "darknet.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yolo_lib.h>
#include <sys/time.h>

using namespace cv;

static image **alphabet;

static char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


static void rgb2yuv422(image *src, void *fb);

static network *net;
void setup_yolo_env(char *cfgfile, char *weightfile)
{
	dector_printf("setup yolo_env\n");

	net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
	alphabet = load_alphabet();

}
#if 0
void set_cur_img_by_name(char *filename)
{
	char *p;
	net->cur_im = load_image_color(filename,0,0);

	strncpy(net->cur_im.name, filename, 256);
	return;
}
#endif
static float *image_normalization(void *p, int w, int h, int c)
{
	int i,j,k;
	char *char_data = (char *)p;

	float *data = (float *)malloc(sizeof(float)*w*h*c);
	for(k = 0; k < c*h*w; ++k){
                data[k] = (float)char_data[k]/255.;
    }    

	return data;
}
#if 0
void set_cur_img(unsigned char *data, int w, int h, int c, const char *name)
{
	net->cur_im.data= (float *)data;
	net->cur_im.w = w;
	net->cur_im.h = h;
	strncpy((net->cur_im.name), name, 256);
	return;
}
#endif
#if 0
float *yolo_inference(float thresh)
{

    clock_t time;
    layer l = net->layers[net->n-1];

	image sized = resize_image(net->cur_im, net->w, net->h);
	float *X = sized.data;
	time=clock();
	network_predict(net, X);
	dector_printf("%s: Predicted in %f seconds.\n", net->cur_im.name, sec(clock()-time));
	return l.output;
}
#endif

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
	void *framebuffer;
	size_t ret;

    if(argc < 6){
        fprintf(stderr, "usage: ./darknet zcu102 [input] [input resolution]\n");
        fprintf(stderr, "ex: ./darknet zcu102 resource/yuyv422_640480_car.yuv  640x480\n");
    }



    width = atoi(strtok((char *)res, "x"));
    height = atoi(strtok(NULL, "x"));
	dector_printf("%s, %dx%d\n", file, width,height);

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

    dector_printf("read %ld bytes, error %d\n", ret, ferror (fp));
    fclose(fp);

    setup_yolo_env(argv[4], argv[5]);

#ifdef NNPACK
    nnp_initialize();
	net->threadpool = pthreadpool_create(2);
#endif

	framebuffer = malloc(f_size);
    yolo_inference_with_ptr(yuv422_data, width, height, 2, 0.2, framebuffer);
#ifdef NNPACK
	pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif

}


static void image_denormalize(image *im)
{
	int i = 0;
	void *v;
	char *p;

	p = (char *)im->data;
	for(i = 0 ; i < im->h * im->w * im->c ; i++)
		p[i] = (char)(im->data[i] * 255);

	return;
}


int yolo_inference_with_ptr(void *ptr, int w, int h, int c, float thresh, void *fb)
{

    void *yuv422_data;
    void *rgb24_data;
    void *resiz_data;
	clock_t time;
	clock_t overall;
	struct timeval start, stop;

	gettimeofday(&start, 0);
	overall = clock();

	layer l = net->layers[net->n-1];
    float nms=.4;
    box *boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(int j = 0; j < l.side*l.side*l.n; ++j)
		probs[j] = (float *)calloc(l.classes, sizeof(float));


	yuv422_data = ptr;
	Mat YUV422_Mat(h, w, CV_8UC2, yuv422_data);

	rgb24_data = malloc(h*w*3);
	Mat RGB24_Mat(h ,w ,CV_8UC3,rgb24_data);
	cvtColor(YUV422_Mat, RGB24_Mat, COLOR_YUV2RGB_YUYV);
	dector_printf("RGB conversion point %fs.\n", sec(clock()-overall));

	resiz_data = malloc(net->h*net->w*net->c);
	Mat Resize_Mat(net->h, net->w,CV_8UC3, resiz_data);
	resize(RGB24_Mat, Resize_Mat, Size(net->h, net->w));
	dector_printf("resize point %fs.\n", sec(clock()-overall));

#ifdef DEBUG
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
	/*FIXME why new need to check
	* Make resized to be planar in order to go through the network.
	*/
	IplImage *img = new IplImage(Resize_Mat);
	image resized = ipl_to_image(img);
	dector_printf("planarlized point %fs.\n", sec(clock()-overall));
	/*original_image keeps to interleaved, since it is used to draw the box directly*/
	image original_image;
	original_image.w = w ; 
	original_image.h = h ;
	original_image.c = 3 ;
	original_image.data = image_normalization(RGB24_Mat.ptr(), original_image.w, original_image.h, original_image.c);
	original_image.c_type = INTERLEAVED;
	dector_printf("pre-processing takes %f seconds.\n", sec(clock()-overall));

	time=clock();
	network_predict(net, resized.data);
	dector_printf("Predicted in %f seconds.\n", sec(clock()-time));

	get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0); 

	dector_printf("get box point %fs.\n", sec(clock()-overall));
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms); /*eliminate the similar box and only left 1 box*/
	dector_printf("nms point %fs.\n", sec(clock()-overall));
	
	draw_detections(original_image, l.side*l.side*l.n, thresh, boxes, probs, 0, voc_names, alphabet, 20);

	dector_printf("draw box point %fs.\n", sec(clock()-overall));

	image_denormalize(&original_image);
	rgb2yuv422(&original_image, fb);

	printf("overall time in %f seconds.\n", sec(clock()-overall));
	gettimeofday(&stop, 0);
	printf("Predicted in %ld ms.\n", (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000));
#if defined(__ARM_ARCH)
	write_raw_image(fb,"/media/card/fb_yuyv422_640480_car.raw",w*h*2 );
#else
	write_raw_image(fb,"fb_yuyv422_640480_car.raw",w*h*2 );
#endif
#ifdef DEBUG
	/*image should be plannar and normalized*/
	write_raw_image(original_image.data ,"bonding_rgb_640480_car.raw",w*h*3);
	write_raw_image(fb,"fb_yuyv422_640480_car.raw",w*h*2 );
	write_raw_image(original_image.data ,"bonding_rgb_640480_car.raw",w*h*3);	
#if defined(OPENCV) && !defined(__ARM_ARCH)
    	save_image(resized, "predictions");
    	show_image(resized, "predictions");	
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
#endif
	free_image(original_image);
	free_image(resized);

	return 0;
}

static void set_pixel_inter(image m, int x, int y, int c, float val) 
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
   // m.data[c*m.h*m.w + y*m.w + x] = val; 
	m.data[y*m.w*m.c + x*m.c + m.c] = val; 
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

void draw_label_inter(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;

    if (r - h >= 0) r = r - h; 


    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(label, i, j, k);
                set_pixel_inter(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }    
}


void draw_box_inter(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;

    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[0 + i*a.c + y1*a.w*a.c] = r;
        a.data[0 + i*a.c + y2*a.w*a.c] = r;

        a.data[1 + i*a.c + y1*a.w*a.c] = g;
        a.data[1 + i*a.c + y2*a.w*a.c] = g;

        a.data[2 + i*a.c + y1*a.w*a.c] = b;
        a.data[2 + i*a.c + y2*a.w*a.c] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[0 + x1*a.c + i*a.w*a.c] = r;
        a.data[0 + x2*a.c + i*a.w*a.c] = r;

        a.data[1 + x1*a.c + i*a.w*a.c] = g;
        a.data[1 + x2*a.c + i*a.w*a.c] = g;

        a.data[2 + x1*a.c + i*a.w*a.c] = b;
        a.data[2 + x2*a.c + i*a.w*a.c] = b;
    }
}


static void rgb2yuv422(image *src, void *fb)
{
	int i,j,n_row,n_col;
	int R, G, B;
	int Y, U, V;
	int RGBIndex, YIndex, UVIndex;
	int src_width,src_height,input_size;

	src_width = src->w;
	src_height = src->h;


	input_size = src_width * src_height;

	unsigned char *RGBBuffer = (unsigned char *)src->data;
	unsigned char *yuyvBuffer;
	unsigned char *YBuffer = new unsigned char[input_size];
	unsigned char *UBuffer = new unsigned char[input_size/2];
	unsigned char *VBuffer = new unsigned char[input_size/2];
	unsigned char *ULine = (new unsigned char[src_width+2])+1;
	unsigned char *VLine = (new unsigned char[src_width+2])+1;

	ULine[-1]=ULine[src_width]=128;
	VLine[-1]=VLine[src_width]=128;

	for (i=0; i<src_height; i++)
	{
		RGBIndex = 3*src_width*i;
		YIndex    = src_width*i;
		UVIndex   = src_width*i/2;

		for ( j=0; j<src_width; j++)
		{
			R = RGBBuffer[RGBIndex++];
			G = RGBBuffer[RGBIndex++];
			B = RGBBuffer[RGBIndex++];
			//Convert RGB to YUV
			Y = (unsigned char)( ( 66 * R + 129 * G +   25 * B + 128) >> 8) + 16   ;
			U = (unsigned char)( ( -38 * R -   74 * G + 112 * B + 128) >> 8) + 128 ;
			V = (unsigned char)( ( 112 * R -   94 * G -   18 * B + 128) >> 8) + 128 ;
			YBuffer[YIndex++] = static_cast<unsigned char>( (Y<0) ? 0 : ((Y>255) ? 255 : Y) );
			VLine[j] = V;
			ULine[j] = U;
		}
		for ( j=0; j<src_width; j+=2)
		{
			//Filter line
			V = ((VLine[j-1]+2*VLine[j]+VLine[j+1]+2)>>2);
			U = ((ULine[j-1]+2*ULine[j]+ULine[j+1]+2)>>2);

			//Clip and copy UV to output buffer
			VBuffer[UVIndex] = static_cast<unsigned char>( (V<0) ? 0 : ((V>255) ? 255 : V) );
			UBuffer[UVIndex++] = static_cast<unsigned char>( (U<0) ? 0 : ((U>255) ? 255 : U) );
		}
	}
	YIndex = 0;
	unsigned char *p = (unsigned char *)fb;
	YIndex = UVIndex = 0;	

	for(j=0; j < input_size*2 ; j += 4)
	{
		p[j] = YBuffer[YIndex++];
		p[j+1] = UBuffer[UVIndex];
        p[j+2] = YBuffer[YIndex++];
        p[j+3] = VBuffer[UVIndex++];
	}

	return;
}
