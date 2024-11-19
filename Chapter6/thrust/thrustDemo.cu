#include "mex.h"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

// To compile: mexcuda thrustDemo.cu

float getSum(float* A, int size)
{
    thrust::device_vector<float> deviceA(A, A + size);
    return thrust::reduce(deviceA.begin(),
                          deviceA.end(),
                          (float)0.0f,
	                      thrust::plus<float>());
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgTxt("Invaid number of input arguments");

    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgTxt("input data type must be single");

    float* A = (float*)mxGetData(prhs[0]);
    int numARows = mxGetM(prhs[0]);
    int numACols = mxGetN(prhs[0]);
    int numElem = (numARows > numACols) ? numARows : numACols;

    plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxCOMPLEX);
    float* B = (float*)mxGetData(plhs[0]);

    *B = getSum(A, numElem);
}
