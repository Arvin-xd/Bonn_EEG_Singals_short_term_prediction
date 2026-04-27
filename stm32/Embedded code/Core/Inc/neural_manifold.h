#ifndef __MY_NM_H__
#define __MY_NM_H__

#include <stdbool.h>

#define INPUT_MATRIX_MAX_COL	20
#define INPUT_MATRIX_MAX_ROW	400

#define INPUT_TRAIN_LENGTH (6)
#define PREDICT_LENGTH (3)

#define DYNAMICS_ROW (200)

#if (0 < (INPUT_TRAIN_LENGTH - 3 * PREDICT_LENGTH))
	#error "param error"
#else
	#define TRAIN_LENGTH (6)
#endif /*(0 < (INPUT_TRAIN_LENGTH - 3 * PREDICT_LENGTH))*/

#define USE_MATRIX_B_SIZE 10

#define TOL 1e-6f // float阈值（适配嵌入式精度）

typedef struct {
	unsigned int row; 			/*行数*/
	unsigned int col;			/*列数*/
	unsigned int memory;		/*数据*/
}Matrix;

Matrix* getInputMatrix(void);

void* getMatrixElementPosition(Matrix *m,unsigned int row,unsigned int col);

bool fillTestData(Matrix *matrix);
bool NMRun(Matrix *input,
	void (*callback)(unsigned short off_set,float *prediction,unsigned int prediction_size,float err));

#endif /*__MY_NM_H__*/
