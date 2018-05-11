#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;
#ifdef GPU
    l.forward_gpu = forward_detection_layer_gpu;
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

void forward_detection_layer(const detection_layer l, network net)
{

    int locations = l.side*l.side;
    int i,j;
    /*[Lucas review] l.output = 7*7*30 = class..class(7*7*20),scale..scale(7*7*2),box..box(7*7*2*4)[x,y,w,h] */
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;

    /*[Lucas review] do softmax for classes of grids*/
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    
    if(net.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
	    /* [Lucas review] locations = 7*7 */
            /*[Lucas review] truth[] layout: 1 valid bit, 20 class,x,y,w,h  (total = 7*7*1+20+4)*/
            for (i = 0; i < locations; ++i) {
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = net.truth[truth_index];
	     /*[Lucas review] l.n = the number of predicted box = 3*/
                for (j = 0; j < l.n; ++j) {
		/*[Lucas review] l.output = 7*7*30 = class..class(7*7*20),scale..scale(7*7*2),box..box(7*7*2*4)[x,y,w,h] */
	     /*[Lucas review] p_index = index of confidence score for j'box*/
	     /*[Lucas review] noobject part's confidence score part in loss function*/	    
                    int p_index = index + locations*l.classes + i*l.n + j;
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj){
                    continue;
                }

                int class_index = index + i*l.classes;
	/*[Lucas review] l.delta's layout is the same as 7*7*30*/
        /*[Lucas review] classification part in loss function.  */
                for(j = 0; j < l.classes; ++j) {
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }

                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
		/*[Lucas review] only for reduce calculation complexity to some extent.*/
                truth.x /= l.side;
                truth.y /= l.side;

		/*[Lucas review] l.n = 2 =  the number of box*/
                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);
                   out.x /= l.side;
                   out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
		/*[Lucas review] get the predicted confidence score for the grid */
                int p_index = index + locations*l.classes + i*l.n + best_index;
		/*[Lucas review] reduct extra added confidence loss of grids back on noobj loss accumulation*/
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
		/*[Lucas review] lamda obj part in loss function*/
		/*[Lucas review] only calculate the loss for best box. as metioned in paper*/
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

		/*[Lucas review] corrdination part in loss function*/
                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
		    /*[Lucas review] when l.sqrt=1, l.ouput[w,h] are trainning as sqrt(w) and sqrt(h), otherwise as w and h.*/
                    /*[Lucas review] yolov1.cfg follow the loss function paper described, l.sqrt = 1*/
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }
		/*[Lucas review] FIXME loss function already compensate the loss of confidence score, why */
                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou; /*used to calculated average iou of represented boxes.*/
                ++count; /*[Lucas review] counting the quantity of represented boxes for 7x7 grids.*/
            }
        }

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }

	/*[Lucas review] FIXME don't know what is this line for ?*/
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}

void backward_detection_layer(const detection_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void save_prediction_files(float *data)
{
    int fd;
    int nr;
    unsigned int* dump;
    
    fd = open("prediction_cube.bin", O_CREAT | O_SYNC | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1)
	       printf("[OBJ DECTOR] save data failed\n");

    printf("[OBJ DECTOR] float = %d bytes\n",sizeof(float));
    printf("[OBJ DECTOR] prediction size = %d bytes\n", 7*7*30*sizeof(float));
    nr = write(fd, data,7*7*30*sizeof(float));
    if (nr == -1)
	       printf("[OBJ DECTOR] copy error\n");

    dump = data;
    printf("[OBJ DECTOR] save %d bytes to ./prediction_cube.bin\n", nr);
    printf("[OBJ DECTOR] dump data for check\n");
    printf("000000000: %x %x %x %x\n", *dump, *(dump+1), *(dump+2),*(dump+3));
    printf("000016f0: %x %x\n", *(dump + 1468), *(dump + 1469));
    return ;
}


void get_detection_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;

    save_prediction_files(predictions);

    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
	for(n = 0; n < l.n; ++n){
            int index = i*l.n + n; /*class..class(7*7*20),scale..scale(7*7*2),box..box(7*7*2*4)[x,y,w,h] */
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;

            /*[Lucas review] x,y is calculated and based on 7*7 grids, so it need to divide to l.side */
   	    /*[Lucas review] x,y is the offset withine a grid, so prediction[]+col/raw represents the central corrdination completly */
            boxes[index].x = (predictions[box_index + 0] + col) / l.side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / l.side * h;
            boxes[index].w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;/* a array carries all prob(class)*confidence for each box. e.g.  [box][class], each element is prob(class)*con */
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network net)
{
    if(!net.train){
        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
        return;
    }

    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    forward_detection_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void backward_detection_layer_gpu(detection_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

