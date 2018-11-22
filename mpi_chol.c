# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "mpi.h"
# include "chol.h"

# define MAX_PROCESSOR_NUM 6
# define MAX_ARRAY_SIZE 32

// 函数声明
int check_if_symmetric(const Matrix M);

int check_if_diagonal_dominant(const Matrix M);

Matrix create_positive_definite_matrix(unsigned int num_rows,unsigned int num_columns);

Matrix allocate_matrix(int num_rows ,int num_columns ,int init);

void print_matrix(const Matrix);

Matrix cholesky(Matrix M);

int main(int argc, char *argv[])
{
	printf("生成对角占优正定对称阵A:\n");
	A = create_positive_definite_matrix(MATRIX_SIZE,MATRIX_SIZE);
	print_matrix(A);

	printf("开始MPI多进程cholesky分解:\n");
	printf("---------------------------\n");
	Matrix L;
	L = mpi_chol(A);
	printf("计算结束，cholesky分解结果:\n");
	print_matrix(L);

	free(A.elements);
	free(L.elements);

	return 0;
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

Matrix allocate_matrix(int num_rows,int num_columns,int init)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows;
	int size = M.num_rows * M.num_columns;
	M.elements = (double*) malloc(size * sizeof(double));

	for (unsigned int i=0;i<size;i++)
	{
		if (init == 0) M.elements[i] = 0;
		else 
		M.elements[i] = (double) rand() / (double)RAND_MAX;
	}
	return M;
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

Matrix mpi_chol(Matrix M)
{
	Matrix L = allocate_matrix(M.num_rows,M.num_columns,0)
	int n;
	n = M.num_rows*M.num_columns;
	double transTime = 0,tempCurrentTime,beginTime;

	MPI_Status status;
	int rank,size;
	int i,j,k;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	for (k = 0;k<n;k++)
	{
		//每次都将从第k个开始往后的数据分配给其他的进程
		MPI_Bcast(M.elements[k*M.num_rows+k],(n-k)*M.num_columns,MPI_DOUBLE,0,MPI_COMM_WORLD);
		for (i=k+rank;i<n;i+=size)
		{
			for (j=0;j<k;j++)
			{
				L.elements[i*M.num_rows+j] = M.elements[i*M.num_rows+j];
			}
			if (i==k)
			{
				for(j=k;j<n;j++) L.elements[i*M.num_rows+j] = M.elements[i*M.num_rows+j];
			}
			else
			{
				L.elements[i*M.num_rows+j] = M.elements[i*M.num_rows+k]/sqrt(M.elements[k*num_rows+k]);
				for(j=k+1;j<n;j++) L.elements[i*M.num_rows+j] = M.elements[i*M.num_columns+k]*M.elements[k*num_rows+j]/M.elements[k*M.num_columns+k];
			}
		}
	
		for(i=k+rank;i<n;i++)
		{
			MPI_Send(L.elements[i*M.num_rows+j],M.num_columns,MPI_DOUBLE,0,k*1000+i,MPI_COMM_WORLD);		
		}
	
		if (rank==0)
		{
			printf("mpi_chol分解结果:\n");
			print_matrix(L);	
		}
	}
	MPI_Finalize();
}