/*
 * gemm.c
 *
 *  Created on: 2018年5月17日
 *      Author: lucas
 */

#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdio.h>
#include "gemm_x86.h"
using namespace Eigen;

typedef Matrix<float,Dynamic,Dynamic,RowMajor> MatrixType;
typedef Map<MatrixType> MapType;

void eigen_gemm(float *a, unsigned int ar, unsigned int ac,
		float *b, unsigned int br, unsigned int bc, float *c)
{
	MapType w_mat(a, ar,ac);
	MapType input_mat(b,br,bc);
	MapType output(c, ar,bc);


	output = w_mat * input_mat;
	return;
}
