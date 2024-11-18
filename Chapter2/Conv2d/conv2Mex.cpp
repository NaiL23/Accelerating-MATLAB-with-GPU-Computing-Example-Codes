#include "mex.h"
// To compile: mex conv2Mex.cpp

void conv2Mex(float* src, float* dst, int numRows, int numCols, float* kernel)
{
    int boundCol = numCols - 1;
    int boundRow = numRows - 1;

    for (int c = 1; c < boundCol; c++)
    {
        for (int r = 1; r < boundRow - 1; r++)
        {
            int dstIndex = c * numRows + r;
            int kerIndex = 8;
            for (int kc = -1; kc < 2; kc++)
            {
                int srcIndex = (c + kc) * numRows + r;
                for (int kr = -1; kr < 2; kr++)
                    dst[dstIndex] += kernel[kerIndex--] * src[srcIndex + kr];
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgTxt("Invaid number of input arguments");
    
    if (nlhs != 1)
        mexErrMsgTxt("Invalid number of outputs");
    
    if (!mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1]))
        mexErrMsgTxt("input image and kernel type must be single");
    
    float* image = (float*)mxGetData(prhs[0]);
    float* kernel = (float*)mxGetData(prhs[1]);
    
    int numRows = mxGetM(prhs[0]);
    int numCols = mxGetN(prhs[0]);
    int numKRows = mxGetM(prhs[1]);
    int numKCols = mxGetN(prhs[1]);

    if (numKRows != 3 || numKCols != 3)
        mexErrMsgTxt("Invalid kernel size. It must be 3x3");
    
    plhs[0] = mxCreateNumericMatrix(numRows, numCols, mxSINGLE_CLASS, mxREAL);
    float* out = (float*)mxGetData(plhs[0]);

    conv2Mex(image, out, numRows, numCols, kernel);
}