GPU=0
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=0
EIGEN=1
QUANTIZE=0
PROF=1
ARCH_X86=1
ARCH_ARM=0
THREADED_EIGEN=0
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
COMMON= -Iinclude/ -Isrc/ -Iplugin/
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(ARCH_X86),1)
CC=gcc
CCP=g++
LDFLAGS= -lm -pthread
endif

ifeq ($(ARCH_ARM),1)
CC=/home/lucas/Project/ZCU102/source/bsp/petalinux_2017-4/tools/linux-i386/aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc
CCP=/home/lucas/Project/ZCU102/source/bsp/petalinux_2017-4/tools/linux-i386/aarch64-linux-gnu/bin/aarch64-linux-gnu-g++
LDFLAGS= -L/home/lucas/Project/ZCU102/source/bsp/petalinux_2017-4/tools/linux-i386/aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/lib/ -lm -pthread
endif



ifeq ($(PROF),1)
CFLAGS+= -DPROFILING
endif

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda-8.0/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda-8.0/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o attention.o darknet.o

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

ifeq ($(EIGEN), 1) 
LDFLAGS+= -lstdc++
OBJ+=vec_gemm.o
CFLAGS+= -DEIGEN
ifeq ($(THREADED_EIGEN),1)
CFLAGS+= -DTHREADED_EIGEN
endif
endif

ifeq ($(QUANTIZE),1) 
CFLAGS+= -DQUANTIZATION
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h


#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS) $(CPPOBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CCP) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)

