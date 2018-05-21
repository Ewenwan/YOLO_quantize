#include "darknet.h"


static network *net;
void setup_yolo_env(char *cfgfile, char *weightfile)
{
	printf("setup yolo_env\n");
	net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
}

void set_cur_img_by_name(char *filename)
{
	net->cur_im = load_image_color(filename,0,0);
	strncpy(&net->cur_im.name, filename, 256);
	return;
}

static void norm_cur_img(void)
{
	return;
}

void set_cur_img(unsigned char *data, int w, int h, int c, char *name)
{
	net->cur_im.data= (float *)data;
	net->cur_im.w = w;
	net->cur_im.h = h;
	strncpy(&(net->cur_im.name), name, 256);
	norm_cur_img();
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



