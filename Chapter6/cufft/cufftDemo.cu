#include "mex.h"
#include <cufft.h>

// To compile: mexcuda cufftDemo.cu -lcufft

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgTxt("Invaid number of input arguments");

    if (!mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1]))
        mexErrMsgTxt("input data type must be single");

    float* A = (float*)mxGetData(prhs[0]);

    int numARows = mxGetM(prhs[0]);
    int numACols = mxGetN(prhs[0]);

    float *deviceA;

    cudaMalloc(&deviceA, sizeof(float) * numARows * numACols);
    cudaMemcpy(deviceA, A, numARows * numACols * sizeof(float),
               cudaMemcpyHostToDevice);
 
    int outRows = numARows /2 + 1;
    int outCols = numACols;
    cufftComplex* deviceOut;
    cudaMalloc(&deviceOut, sizeof(cufftComplex) * outRows * outCols);

    cufftHandle plan;
    cufftPlan2d(&plan, numACols, numARows, CUFFT_R2C);
    cufftExecR2C(plan, deviceA, deviceOut);

    float* out = (float*)mxMalloc(sizeof(cufftComplex) * outRows * outCols);
    cudaMemcpy(out, deviceOut, outRows * outCols * sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);

    plhs[0] = mxCreateNumericMatrix(outRows, outCols, mxSINGLE_CLASS, mxCOMPLEX);
    float* real = (float*)mxGetPr(plhs[0]);
    float* imag = (float*)mxGetPi(plhs[0]);
    float* complex = out;
    for (int c = 0; c < outCols; ++c)
    {
        for (int r = 0; r < outRows; ++r)
        {
            *real++ = *complex++;
            *imag++ = *complex++;
        }
    }

    mxFree(out);
    cufftDestroy(plan);
    cudaFree(deviceA);
    cudaDeviceReset();
}
