#include <opencv2/opencv.hpp>
using namespace cv;

#define INPUT_RAW_PIXEL	(768)
#define INPUT_COL_PIXEL	(576)
#define INPUT_SIZE	(INPUT_RAW_PIXEL*INPUT_COL_PIXEL)
#define INPUT_RGB_CHANNEL	3
#define OUTPUT_RAW_PIXEL	INPUT_RAW_PIXEL
#define OUTUT_COL_PIXEL		INPUT_COL_PIXEL
#define OUTPUT_LOCATION_X	(400)
#define OUTPUT_LOCATION_Y	(500)
#define PREDICT_CUBE_SIZE	(7*7*30*4)
#define PREDICT_CUBE_WIDTH	(7)
#define PREDICT_CUBE_HEIGHT	(7)
#define PREDICT_CUBE_DEPTH	(30)
#define PREDICT_NUM_OF_BOX  (2)
#define PREDICT_NUM_OF_GRIDS	(PREDICT_CUBE_WIDTH * PREDICT_CUBE_HEIGHT)
#define BOX_THRESH	(0.2)
#define NRM_THRESH	(0.4)
#define PREDICT_CUBE_CENTER_W_SCALE (1)
#define PREDICT_CUBE_CENTER_H_SCALE (1)
#define PREDICT_CUBE_SIZE_SQUARE (1)
#define DEBUG_PRINT (1)
#define HEADER "[OBJ_DECTOR]: "


static void rgb2yuv422(Mat *src, Mat *dst)
{
#if 1
	int i,j,n_row,n_col;
	int R, G, B;
	int Y, U, V;
	int RGBIndex, YIndex, UVIndex;
	unsigned char *RGBBuffer = src->ptr(0);
	unsigned char *yuyvBuffer;
	unsigned char *YBuffer = new unsigned char[INPUT_SIZE];
	unsigned char *UBuffer = new unsigned char[INPUT_SIZE/2];
	unsigned char *VBuffer = new unsigned char[INPUT_SIZE/2];
	unsigned char *ULine = (new unsigned char[INPUT_RAW_PIXEL+2])+1;
	unsigned char *VLine = (new unsigned char[INPUT_RAW_PIXEL+2])+1;

	ULine[-1]=ULine[512]=128;
	VLine[-1]=VLine[512]=128;

	for (i=0; i<INPUT_COL_PIXEL; i++)
	{
		RGBIndex = 3*INPUT_RAW_PIXEL*i;
		YIndex    = INPUT_RAW_PIXEL*i;
		UVIndex   = INPUT_RAW_PIXEL*i/2;

		for ( j=0; j<INPUT_RAW_PIXEL; j++)
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
		for ( j=0; j<INPUT_RAW_PIXEL; j+=2)
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
	UVIndex = 0;

	for (n_row = 0; n_row < OUTUT_COL_PIXEL; n_row++)
	{
		yuyvBuffer = dst->ptr(OUTPUT_LOCATION_X+2*n_row, OUTPUT_LOCATION_Y);//(row(h),col(p))

		for(n_col = 0; n_col < OUTPUT_RAW_PIXEL; n_col+=2)
		{
			yuyvBuffer[n_col*2 + 0] = YBuffer[YIndex++];
			yuyvBuffer[n_col*2 + 1] = UBuffer[UVIndex];
			yuyvBuffer[n_col*2 + 2] = YBuffer[YIndex++];
			yuyvBuffer[n_col*2 + 3] = VBuffer[UVIndex++];
		}
	}
	delete [] YBuffer;
	delete [] UBuffer;
	delete [] VBuffer;
	delete [] (ULine-1);
	delete [] (VLine-1);

	return;

#endif
}
