#include "tools.hpp"
# define n 6 // length of x

int main( void)
{

	printf("test_link = %i\n", test_link);
	float a,b;

	a =1.0;
	b =1.0;

	std::pair<float, float> c_s = constructGivensRotGPU<float>(a,b);
	
	printf("components of rot matrix are: c = %f,  d = %f\n", c_s.first ,c_s.second);

	return EXIT_SUCCESS;
}


	/* 
	####################   
	Example Min
	####################
	float *x;
	x = createVecCPU<float>(n);
	for(int j=0;j<n;j++)
		x[j]=( float )(j);	

	float min_element;
	int index;
	std::pair<float, int> min_index = findMinGPU<float>(x, 1, n);
	printf("min_element = %f\n", min_index.first);
	*/



	/*
	####################   
	Example Max
	####################
	float *x;
	x = createVecCPU<float>(n);
	for(int j=0;j<n;j++)
		x[j]=( float )(j);	

	
	std::pair<float, int> max_index = findMaxGPU<float>(x, 1, n);
	printf("max_element = %f\n", max_index.first);
	printf("index_of_max_element = %i\n", max_index.second);
	*/

	
	/*
	####################   
	Example Absolute Sum of Array
	####################
	float *x;
	x = createVecCPU<float>(n);
	for(int j=0;j<n;j++)
		x[j]=( float )(j);	

	
	float abs_sum = findAbsSumGPU<float>(x, 1, n);
	printf("abs_sum = %f\n", abs_sum);
	*/

	/*
	####################   
	Example add array y to array x
	####################
	float *x; // n- vector on the host
	float *y; // n- vector on the host
	float *z;

	x = (float*) malloc(n * sizeof(*x)); // host memory alloc for x
	for(int j=0;j<n;j++)
		x[j]=( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}

	y = (float*) malloc(n * sizeof(*y)); // host memory alloc for y
	for(int j=0;j<n;j++)
		y[j]=( float )j; // y={0 ,1 ,2 ,3 ,4 ,5}
	printf ("\n");

	z = (float*) malloc(n * sizeof(*z)); // host memory alloc for z	
	float a = 2.0;

	
	computeAxpyGPU<float>(x, y, a, z, 1, n);
	
	for(int j=0;j<n;j++)
		printf("z[%i] = %f\n", j, z[j]); 
	*/


	/*
	#############################   
	Example make a copy of array x
	#############################
	float *x; // n- vector on the host
	float *y; // n- vector on the host
	
	x = (float*) malloc(n * sizeof(*x)); // host memory alloc for x
	for(int j=0;j<n;j++)
		x[j]=( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}

	y = (float*) malloc(n * sizeof(*y)); // host memory alloc for y
	
	
	copyXtoYGPU<float>(x, y, 1, n);
	
	for(int j=0;j<n;j++)
		printf("x[%i] = %f,  y[%i] = %f\n", j, x[j], j, y[j]); 


	free(x);
	free(y);
	*/


	/*
	#############################   
	Example compute dot product of two vectors
	#############################
	float *x; // n- vector on the host
	float *y; // n- vector on the host
	float z;

	x = (float*) malloc(n * sizeof(*x)); // host memory alloc for x
	for(int j=0;j<n;j++)
		x[j]=( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}

	y = (float*) malloc(n * sizeof(*y)); // host memory alloc for y
	for(int j=0;j<n;j++)
		y[j]=( float )j; // y={0 ,1 ,2 ,3 ,4 ,5}
	printf ("\n");

	

	z = computeXdotYGPU<float>(x, y, 1, n);
	
	printf("Dot product of z and y is %f\n", z);

	free(x);
	free(y);
	*/


	/*
	#############################   
	Example euclidean norm of vector x
	#############################
	float *x; // n- vector on the host
	float z;

	x = (float*) malloc(n * sizeof(*x)); // host memory alloc for x
	for(int j=0;j<n;j++)
		x[j]=( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}

	

	

	z = computeNormEuclidGPU<float>(x, 1, n);
	
	printf("Euclidean norm of vector x is %f\n", z);

	free(x);
	*/



	/*
	#############################   
	Example compute Givens Rotation on a 2_x_n matrix (first row is vector X, second is vector Y
	Rotation matrix is [c  s]
			   [-s c]
	#############################
	float *x; // n- vector on the host
	float *y;

	x = createVecCPU<float>(n);
	for(int j=0;j<n;j++)
		x[j]=( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}

	y = createVecCPU<float>(n);
	for(int j=0;j<n;j++)
		y[j]=( float )(j*j); // x={0 ,1 ,2 ,3 ,4 ,5}

	float c =0.5;
	float s =0.8669254;

	computeGivensRotGPU<float>(x, y, c, s, 1, n);
	
	for(int j=0;j<n;j++)
		printf("x[%i] = %f,  y[%i] = %f\n", j, x[j], j, y[j]); 

	free(x);
	free(y);
	*/
	

	/*
	############################# 
	Example
	constructs the Givens rotation matrix G = [c  s]
						  [-s c]
		that zeros out the 2 x 1 vector [a]
						[b]
		
		such that [c   s] [a] = [r]
			  [-s  c] [b]   [0]
		where, 
			c^2 + s2 = 1; r^2 = a^2 + b^2:
	#############################

	*/
