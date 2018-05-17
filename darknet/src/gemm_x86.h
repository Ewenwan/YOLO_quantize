/*
 * gemm_x86.h
 *
 *  Created on: 2018年5月18日
 *      Author: lucas
 */

#ifndef GEMM_X86_H_
#define GEMM_X86_H_

#ifdef __cplusplus
extern "C" {
#endif
void eigen_gemm(float *a, unsigned int ar, unsigned int ac,
		float *b, unsigned int br, unsigned int bc, float *c);

#ifdef __cplusplus
}
#endif


#endif /* GEMM_X86_H_ */
