//为了防止头文件被重复引用加上宏
# ifndef _MATRIX_H
# define _MATRIX_H

//定义矩阵大小，同时由于是按照行列对矩阵划分的，所以这里的矩阵行列大小也可以放映线程块的大小
# define MATRIX_SIZE 5

//定义矩阵不同维度上的长度
# define NUM_COLUMNS MATRIX_SIZE
# define NUM_ROWS MATRIX_SIZE

//定义矩阵结构体，这里为了方便选择使用数组结构体而不是结构体数组
typedef struct 
{
	//定义矩阵行列维度
	unsigned int num_columns;
	unsigned int num_rows;

	// 定义相邻两行开始元素在内存视图中相隔的元素数量
	unsigned int pitch;
	
	//指向矩阵第一个元素的指针
	double* elements;
} Matrix;

# endif 
