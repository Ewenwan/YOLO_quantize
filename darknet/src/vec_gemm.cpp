/*
 * gemm.c
 *
 *  Created on: 2018年5月17日
 *      Author: lucas
 */

#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdio.h>
#include "vec_gemm.h"
using namespace Eigen;

typedef Matrix<float,Dynamic,Dynamic,RowMajor> MatrixType;
typedef Map<MatrixType> MapType;

#define CHECK_ASM

void eigen_gemm(float *a, unsigned int ar, unsigned int ac,
		float *b, unsigned int br, unsigned int bc, float *c)
{
	MapType w_mat(a, ar,ac);
	MapType input_mat(b,br,bc);
	MapType output(c, ar,bc);
#ifdef CHECK_ASM
	EIGEN_ASM_COMMENT("#it begins here!");
#endif
	output = w_mat * input_mat;
#ifdef CHECK_ASM
 	EIGEN_ASM_COMMENT("#it ends here!");
#endif	
	return;
}

void eigen_vectorize_status(void)
{
        #ifdef EIGEN_VECTORIZE
        printf("[EIGEN] VECTORIZE ENABLE\n");
        #else
        printf("[EIGEN] VECTORIZE NOT ENABLE\n");
        #endif
}
