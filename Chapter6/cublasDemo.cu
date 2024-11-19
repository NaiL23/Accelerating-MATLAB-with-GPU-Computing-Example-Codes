#include "mex.h"
#include <cublas_v2.h>
// To compile: mexcuda cublasDemo.cu -lcublas

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgTxt("Invaid number of input arguments");
   
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgTxt("input matrices must be single");
    
    float* A = (float*)mxGetData(prhs[0]);
    float* B = (float*)mxGetData(prhs[1]);
    
    int numARows = mxGetM(prhs[0]);
    int numACols = mxGetN(prhs[0]);
    int numBRows = mxGetM(prhs[1]);
    int numBCols = mxGetN(prhs[1]);
    int numCRows = numARows;
    int numCCols = numBCols;
    
    plhs[0] = mxCreateNumericMatrix(numCRows, numCCols, mxSINGLE_CLASS, mxREAL);
    float* C = (float*)mxGetData(plhs[0]);
    
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, sizeof(float) * numARows * numACols);
    cudaMalloc(&deviceB, sizeof(float) * numBRows * numBCols);
    cudaMalloc(&deviceC, sizeof(float) * numCRows * numCCols);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMatrix(numARows,
                    numACols,
                    sizeof(float),
                    A,
                    numARows,
                    deviceA,
                    numARows);
    cublasSetMatrix(numBRows,
                    numBCols,
                    sizeof(float),
                    B,
                    numBRows,
                    deviceB,
                    numBRows);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                numARows,
                numBCols,
                numACols,
                &alpha,
                deviceA,
                numARows,
                deviceB,
                numBRows,
                &beta,
                deviceC,
                numCRows);
    
    cublasGetMatrix(numCRows,
                    numCCols,
                    sizeof(float),
                    deviceC,
                    numCRows,
                    C,
                    numCRows);
    
    cublasDestroy(handle);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}