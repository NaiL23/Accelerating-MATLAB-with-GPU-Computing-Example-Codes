// To compile: mex column_major_order_single.cpp

#include "mex.h"
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (!mxIsSingle(prhs[0]))
        mexErrMsgTxt("input vector data type must be single");
  
    int rows = (int)mxGetM(prhs[0]);
    int cols = (int)mxGetN(prhs[0]);
    int totalElements = rows * cols;
 
    float* A = (float*)mxGetData(prhs[0]);
    
    for (int i = 0; i < totalElements; ++i)
        mexPrintf("%f at %p\n", A[i], A + i);
}
