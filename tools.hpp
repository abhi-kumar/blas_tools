#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>

int test_link = 2;

//Create vector on CPU memory
template <class T >
T* createVecCPU(int num_elem)
{
	T *x; 
	x=( T *) malloc (num_elem* sizeof (*x)); 
	return x;
}


/*
	1. Transfer array from cpu to gpu
	2. Find minimum on GPU
	3. Transfer output to CPU from GPU

*/
template <class T >
std::pair <T,int> findMinGPU(T *x, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	T *d_x; 
	cudaStat = cudaMalloc (( void **)& d_x ,num_elem* sizeof (*x)); 

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x) ,x ,increment ,d_x , increment); // cp x ->d_x
	int result ; 	
	stat=cublasIsamin(handle,num_elem,d_x,increment,&result);
	cudaFree (d_x ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	return std::pair<T, int>(x[result-1],result-1);
	//return std::make_pair(x[result-1],result-1);
}



/*
	1. Transfer array from cpu to gpu
	2. Find maximum on GPU
	3. Transfer output to CPU from GPU

*/
template <class T >
std::pair <T,int> findMaxGPU(T *x, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	T *d_x; 
	cudaStat = cudaMalloc (( void **)& d_x ,num_elem* sizeof (*x)); 

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x) ,x ,increment ,d_x ,increment); // cp x ->d_x
	int result ; 	
	stat=cublasIsamax(handle,num_elem,d_x,increment,&result);
	cudaFree (d_x ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1);
	return std::pair<T, int>(x[result-1],result-1);
}



/*
	1. Transfer array from cpu to gpu
	2. Find absolute sum on GPU
	3. Transfer output to CPU from GPU

*/
template <class T >
T findAbsSumGPU(T *x, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	T *d_x; 
	cudaStat = cudaMalloc (( void **)& d_x ,num_elem* sizeof (*x)); 

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x) ,x ,increment ,d_x ,increment); // cp x ->d_x
	float result ; 	
	stat = cublasSasum(handle,num_elem,d_x,increment,&result);
	cudaFree (d_x ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1);
	return result;
}


/*
	1. Transfer arrays from cpu to gpu
	2. Operation z[i] = x[i]*a+y[i] on GPU
	3. Transfer output to CPU from GPU

*/
template <class T >
void computeAxpyGPU(T *x, T *y, T a, T *z, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	float *d_x; // d_x - x on the device
	float *d_y; // d_y - y on the device
	cudaStat = cudaMalloc((void**) &d_x, num_elem * sizeof(*x)); // device
	// memory alloc for x
	cudaStat = cudaMalloc((void**) &d_y, num_elem * sizeof(*y)); // device
	// memory alloc for y

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x), x, increment, d_x, increment); // cp x- >d_x
	stat = cublasSetVector (num_elem, sizeof (*y), y, increment, d_y, increment); // cp y- >d_y

	
	stat = cublasSaxpy(handle,num_elem,&a,d_x,increment,d_y,increment);
	stat = cublasGetVector(num_elem, sizeof ( float ) ,d_y ,increment ,z ,increment); // cp d_y - >y	


	cudaFree (d_x ); // free device memory
	cudaFree (d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1);
}

/*
	1. Transfer arrays from cpu to gpu
	2. Copy array x into array y
	3. Transfer output to CPU from GPU

*/
template <class T >
void copyXtoYGPU(T *x, T *y, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	float *d_x; // d_x - x on the device
	float *d_y; // d_y - y on the device
	cudaStat = cudaMalloc((void**) &d_x, num_elem * sizeof(*x)); // device
	// memory alloc for x
	cudaStat = cudaMalloc((void**) &d_y, num_elem * sizeof(*y)); // device
	// memory alloc for y

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x), x, increment, d_x, increment); // cp x- >d_x
	
	stat = cublasScopy(handle,num_elem,d_x,increment,d_y,increment);
	stat = cublasGetVector (num_elem, sizeof ( float ),d_y ,increment,y ,increment); 	


	cudaFree (d_x ); // free device memory
	cudaFree (d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1)
}



/*
	1. Transfer arrays from cpu to gpu
	2. Operation z = summationOF(x[i].y[i]) for all i, on GPU
	3. Transfer output to CPU from GPU

*/
template <class T >
T computeXdotYGPU(T *x, T *y, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	float *d_x; // d_x - x on the device
	float *d_y; // d_y - y on the device
	cudaStat = cudaMalloc((void**) &d_x, num_elem * sizeof(*x)); // device
	// memory alloc for x
	cudaStat = cudaMalloc((void**) &d_y, num_elem * sizeof(*y)); // device
	// memory alloc for y

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x), x, increment, d_x, increment); // cp x- >d_x
	stat = cublasSetVector (num_elem, sizeof (*y), y, increment, d_y, increment); // cp y- >d_y

	float result;
	stat=cublasSdot(handle,num_elem,d_x,increment,d_y,increment,&result);	


	cudaFree (d_x ); // free device memory
	cudaFree (d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1);
	return result;
}


/*
	1. Transfer arrays from cpu to gpu
	2. compute euclidean norm of a vector on GPU
	3. Transfer output to CPU from GPU

*/
template <class T >
T computeNormEuclidGPU(T *x, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	float *d_x; // d_x - x on the device
	cudaStat = cudaMalloc((void**) &d_x, num_elem * sizeof(*x)); // device
	// memory alloc for x
	

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x), x, increment, d_x, increment); // cp x- >d_x

	float result;
	stat=cublasSnrm2(handle,num_elem,d_x,increment,&result);	


	cudaFree (d_x ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1);
	return result;
}


/*
	1. Transfer arrays from cpu to gpu
	2. Computes givens rotation to a 2_x_num_elem matrix x;
	3. Transfer output to CPU from GPU

*/
template <class T >
void computeGivensRotGPU(T *x, T *y, float c, float s, int increment, int num_elem)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	float *d_x; // d_x - x on the device
	cudaStat = cudaMalloc((void**) &d_x, num_elem * sizeof(*x)); // device
	// memory alloc for x

	float *d_y; // d_y - y on the device
	cudaStat = cudaMalloc((void**) &d_y, num_elem * sizeof(*y)); // device
	// memory alloc for y
	

	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (num_elem, sizeof (*x), x, increment, d_x, increment); // cp x- >d_x
	stat = cublasSetVector (num_elem, sizeof (*y), y, increment, d_y, increment); // cp x- >d_x
	
	stat=cublasSrot(handle,num_elem,d_x,increment,d_y,increment,&c,&s);
	stat = cublasGetVector (num_elem, sizeof ( float ),d_x ,increment,x ,increment); 
	stat = cublasGetVector (num_elem, sizeof ( float ),d_y ,increment,y ,increment); 


	cudaFree (d_x ); // free device memory
	cudaFree (d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1)
}


template <class T >
std::pair <T,T> constructGivensRotGPU(T a, T b)
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	T c, s;	
	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSrotg(handle,&a,&b,&c,&s);

	cublasDestroy ( handle ); // destroy CUBLAS context
	//printf("%f    %i", x[result-1], result-1)
	return std::pair<T, T>(c, s);
}


