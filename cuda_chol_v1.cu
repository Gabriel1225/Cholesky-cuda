# include<stdio.h>
# include<cstdio>
# include<cmath>
# include<cstdlib>
# include<ctime>
# include<time.h>
# include<cuda_runtime.h>
# include "chol.h"

// CUDA初始化
bool initCUDA();

// 检查是否是对角占优阵
int check_if_diagonal_dominant(const Matrix M);

// 建立由随机数组成的正定矩阵
Matrix create_positive_definite_matrix(unsigned int,unsigned int);

// host和device之间的数据传输
void copy_matrix_to_device(Matrix Mdevice,const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost,const Matrix Mdevice);

// 在GPU上初始化cholesky矩阵
Matrix allocate_matrix_on_gpu(const Matrix M);

// A被cholesky分解
Matrix A;
Matrix h_A;

// 输出矩阵元素
void print_matrix(const Matrix);

// 检查矩阵是否对称

int check_if_symmetric(const Matrix M);

// 这里选择的是输出上三角矩阵L
__global__ void chol_kernel(double * U,int ops_per_thread)
{
	// const int tid = threadIdx.x;
	// const int size = U.num_rows / THREAD_NUM;
	// Matrix L = allocate_matrix_on_gpu(U);
	// for (unsigned int i=tid*size;i<(tid+1)*size;i++)
	// {
	// 	for (unsigned int j=0;j<(i+1);j++)
	// 	{
	// 		double s = 0;
	// 		for (unsigned int k=0;k<j;k++)
	// 			s += L.elements[i*U.num_rows+k]*L.elements[j*U.num_rows+k];
	// 		L.elements[i*U.num_rows+j] = (i==j) ? sqrt(U.elements[i*U.num_rows+i]-s) : (1.0/L.elements[j*U.num_rows+j]*(U.elements[i*U.num_rows+j]-s));
	// 	}
	// }
	// return L;


	//int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	unsigned int i,j,k;
	unsigned int num_rows = MATRIX_SIZE;
	for (k =0;k<num_rows;k++)
	{
		if (tx == 0)
		{
			U[k*num_rows+k] = sqrt(U[k*num_rows+k]);
			for (j=(k+1);j<num_rows;j++)
			{
				U[k * num_rows + j] /=  U[k * num_rows + k];
			}
		}
		__syncthreads();

		int istart = (k+1) + tx*ops_per_thread;
		int iend = istart + ops_per_thread;
		for (i=istart;i<iend;i++)
		{	
			for (j=i;j<num_rows;j++)
				{
					U[i*num_rows+j] -= U[k*num_rows+i]*U[k*num_rows+j];
				}
		}
		__syncthreads();
	}
	__syncthreads();

	int istart = tx*ops_per_thread;
	int iend = istart+ops_per_thread;

	for(i=istart;i<iend;i++)
	{
		for (j=0;j<i;j++)
			U[i*num_rows+j] = 0.0;
	}

}


int main()
{
	if (!initCUDA())
		return 0;

	A = create_positive_definite_matrix(MATRIX_SIZE,MATRIX_SIZE);
	print_matrix(A);
	
	//int num_block = 1;
	//int threads_per_block = 512;
	int num_threads = 1;
	float ops_per_thread = MATRIX_SIZE / num_threads;
	//dim3 thread_block(threads_per_block,1,1);
	//dim3 grid(num_block,1);

	Matrix d_A = allocate_matrix_on_gpu(A);
	copy_matrix_to_device(d_A,A);
	chol_kernel<<<1,num_threads,0>>>(d_A.elements,ops_per_thread);
	cudaDeviceSynchronize();
	copy_matrix_from_device(A,d_A);
	print_matrix(A);
	
	cudaFree(d_A.elements);
	free(A.elements);

	return 0;
}

bool initCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if(count==0)
	{
		fprintf(stderr, "不好，没有可用的设备!\n");
		return false;
	}
	int i;
	for ( i=0;i<count;i++)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop,i)==cudaSuccess)
		{
			if (prop.major >=1)
				break;
		}
	}
		if (i == count)
		{
			fprintf(stderr, "有设备，但是不支持cuda1.x以上！\n");
			return false;
		}
		cudaSetDevice(i);
		return true;
}


Matrix create_positive_definite_matrix(unsigned int num_rows, unsigned int num_columns)
{
	// 配置矩阵结构体并分配元素地址空间
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows;
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (double*)malloc(size * sizeof(double));

	printf("正在生成 %d x %d 元素大小在正负.5之间的矩阵\n",num_rows,num_columns);
	unsigned int i;
	unsigned int j;
	for (i=0;i<size;i++)
	{
		M.elements[i] = ((double)rand()/(double)RAND_MAX)-0.5;		
	}
	printf("随机矩阵生成完成。\n");
	print_matrix(M);

	//这里使用的方法是原矩阵加上其对称阵
	printf("将矩阵转化为对称阵：\n");
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_rows;
	transpose.num_rows = num_columns;
	size = transpose.num_rows * transpose.num_columns;
	transpose.elements = (double*)malloc(size* sizeof(double));

	for (i=0;i<transpose.num_rows;i++)
		for(j=0;j<transpose.num_columns;j++)
		transpose.elements[i*transpose.num_rows + j] = M.elements[j*M.num_columns + i];

	for(i=0;i<size;i++)
		M.elements[i] += transpose.elements[i];

	printf("检查是否满足对称阵条件.......\n");
	if(check_if_symmetric(M))
		printf("满足对称阵条件\n");
	else
	{
		printf("不满足对称阵条件，程序有误！\n");
		free(M.elements);
		M.elements = NULL;
	}
	free(transpose.elements);

	printf("将对称阵转化为正定矩阵：\n");
	for (i=0;i<num_rows;i++)
		for (j=0;j<num_columns;j++)
		{
			if(i==j) 
				M.elements[i*num_rows + j] += 0.5 * M.num_rows;
		}

	if(check_if_diagonal_dominant(M))
		printf("矩阵是正定阵，满足条件\n");
	else
	{
		printf("矩阵不是正定阵，程序有误\n");
		free(M.elements);
		M.elements = NULL;
	}
	return M;
}


int check_if_diagonal_dominant(const Matrix M)
{
	double diag_element;
	double sum;
	for (unsigned int i=0;i<M.num_rows;i++)
	{	
		diag_element = M.elements[i*M.num_rows+i];
		sum = 0.0;
		for(unsigned int j=0;j<M.num_columns;j++)
		{ 
			if(i != j) sum += abs(M.elements[i*M.num_rows+j]);
		}	
		if (diag_element < sum) return 0; 
	}
	return 1;
}

void copy_matrix_to_device(Matrix Mdevice,const Matrix Mhost)
{
	Mdevice.num_rows = Mhost.num_rows;
	Mdevice.num_columns = Mhost.num_columns;
	Mdevice.pitch = Mhost.pitch;
	int size = Mhost.num_rows * Mhost.num_columns;
	//cudaMalloc((void**)&Mdevice.elements,sizeof(double)* size);
	cudaMemcpy(Mdevice.elements,Mhost.elements,sizeof(double)* size,cudaMemcpyHostToDevice);
	printf("矩阵从host到device传输完成！\n");
}


void copy_matrix_from_device(Matrix Mhost,Matrix Mdevice)
{
	Mhost.num_rows = Mdevice.num_rows;
	Mhost.num_columns = Mdevice.num_columns;
	int size = Mdevice.num_rows*Mdevice.num_columns;
	cudaMemcpy(Mhost.elements,Mdevice.elements,sizeof(double)*size,cudaMemcpyDeviceToHost);
	printf("矩阵从device到host传输完成!\n");
}

Matrix allocate_matrix_on_gpu(const Matrix M)
{
	Matrix L;
	L.num_rows = M.num_rows;
	L.num_columns = L.pitch = M.num_columns;
	int size = L.num_rows * L.num_columns;
	cudaMalloc((void**)&L.elements,sizeof(double)*size);
	return L;
}

void print_matrix(const Matrix M)
{
	for (unsigned int i = 0; i<M.num_rows; i++)
	{
		for(unsigned int j = 0;j<M.num_columns;j++)	
			printf("%f ",M.elements[i*M.num_rows+j]);
	printf("\n");
	}
	printf("\n");
}

int check_if_symmetric(const Matrix M)
{
	for (unsigned int i=0;i<M.num_rows;i++)
		for(unsigned int j=0;j<M.num_columns;j++)
		{
			if (M.elements[i*M.num_rows+j] != M.elements[j*M.num_columns+i])
				return 0;
		}
		return 1;
}









