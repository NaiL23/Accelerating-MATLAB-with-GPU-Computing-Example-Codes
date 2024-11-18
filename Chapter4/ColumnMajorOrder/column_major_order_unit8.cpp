// To compile: mex column_major_order_unit8.cpp

#include "mex.h"
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (!mxIsUint8(prhs[0]))
        mexErrMsgTxt("input vector data type must be uint8");
  
    int rows = (int)mxGetM(prhs[0]);
    int cols = (int)mxGetN(prhs[0]);
    int totalElements = rows * cols;
 
    unsigned char* A = (unsigned char*)mxGetData(prhs[0]);
    
    for (int i = 0; i < totalElements; ++i)
        mexPrintf("%d at %p\n", A[i], A + i);
}
