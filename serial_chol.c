# include <stdio.h>
# include <math.h>
# include <stdlib.h>
# include <time.h>
# include "chol.h"

//声明函数

// 检验矩阵M是否是对角占优矩阵，是对角占优返回1，不是对角占优返回0
int check_if_diagonal_dominant(const Matrix M);

// 给出矩阵大小随机生成对角占优正定对称矩阵
Matrix create_positive_definite_matrix(unsigned int num_rows,unsigned int num_columns);

// 初始化输出的cholesky分解矩阵
Matrix allocate_matrix(int num_rows ,int num_columns ,int init);

// 定义矩阵A
Matrix A;

// 打印矩阵
void print_matrix(const Matrix);

// 检查矩阵是否对称，不对称输出0，对称输出1
int check_if_symmetric(const Matrix M);

// Cholesky分解
Matrix cholesky(Matrix M);

// 定义矩阵大小
//# define MATRIX_SIZE 3


int main()
{	
	printf("生成生成对角占优正定对称阵A：\n");
	A = create_positive_definite_matrix(MATRIX_SIZE,MATRIX_SIZE);
	print_matrix(A);

	printf("开始串行cholesky分解：\n");
	printf("-----------------------\n");
	Matrix L;
	printf("-----------------------\n");
	L = cholesky(A);
	printf("计算结束，cholesky分解结果：\n");
	print_matrix(L);

	free(A.elements);
	free(L.elements);

	return 0;
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

Matrix cholesky(Matrix M)
{
	Matrix L = allocate_matrix(M.num_rows,M.num_columns,0);
	for (unsigned int i=0;i<M.num_rows;i++)
	{
		for(unsigned int j=0;j<(i+1);j++)
			{
				double s = 0;
				for (unsigned int k=0;k<j;k++)
					s += L.elements[i*M.num_rows+k]*L.elements[j*M.num_rows+k];
			L.elements[i*M.num_rows+j] = (i==j) ? sqrt(M.elements[i*M.num_rows+i]-s) : (1.0/L.elements[j*M.num_rows+j] * (M.elements[i*M.num_rows+j]-s));
			}	
	}
	return L;

}

