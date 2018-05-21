#include "darknet.h"
#include <stdlib.h>

static image **alphabet;

static char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


static network *net;
void setup_yolo_env(const char *cfgfile, const char *weightfile)
{
	printf("setup yolo_env\n");
	net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
	alphabet = load_alphabet();
}

void set_cur_img_by_name(const char *filename)
{
	net->cur_im = load_image_color(filename,0,0);
	strncpy(&net->cur_im.name, filename, 256);
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
	strncpy(&(net->cur_im.name), name, 256);
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

int yolo_inference_with_ptr(void *ptr, int w, int h, int c, float thresh)
{
	clock_t time;
	layer l = net->layers[net->n-1];
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(int j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));


	image input;
	input.w = w;
	input.h = h;
	input.c = c;
	input.data = image_normalization(ptr, w, h, c);
	time=clock();
	network_predict(net, input.data);
	printf("%s: Predicted in %f seconds.\n", net->cur_im.name, sec(clock()-time));

	get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0); 
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms); /*eliminate the similar box and only left 1 box*/
	draw_detections(input, l.side*l.side*l.n, thresh, boxes, probs, 0, voc_names, alphabet, 20);
	save_image(input, "predictions");

	free_image(input);

}


