#include "mex.h"
// To compile: mexcuda conv2MexOptA.cu

__global__ void conv2MexCuda(float* src,
                             float* dst,
                             int numRows,
                             int numCols,
                             float* mask)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < 1 || row > numRows - 1)
        return;

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < 1 || col > numCols - 1)
        return;

    int dstIndex = col * numRows + row;
    dst[dstIndex] = 0;
    int mskIndex = 3 * 3 - 1;
    for (int kc = -1; kc < 2; kc++)
    {
        int srcIndex = (col + kc) * numRows + row;
        for (int kr = -1; kr < 2; kr++)
        {
            dst[dstIndex] += mask[mskIndex--] * src[srcIndex + kr];
        }
    }
}

void conv2Mex(float* src, float* dst, int numRows, int numCols, float* msk)
{
    int totalPixels = numRows * numCols;
    float *deviceSrc, *deviceMsk, *deviceDst;

    cudaMalloc(&deviceSrc, sizeof(float) * totalPixels);
    cudaMalloc(&deviceDst, sizeof(float) * totalPixels);
    cudaMalloc(&deviceMsk, sizeof(float) * 3 * 3);

    cudaMemcpy(deviceSrc, src, sizeof(float) * totalPixels, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMsk, msk, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);
    cudaMemset(deviceDst, 0, sizeof(float) * totalPixels);

    dim3 blockSize(16, 16);
    dim3 gridSize((numRows +15) / blockSize.x, (numCols + 15) / blockSize.y);
    conv2MexCuda<<<gridSize, blockSize>>>(deviceSrc,
                                          deviceDst,
                                          numRows,
                                          numCols,
                                          deviceMsk);

    cudaMemcpy(dst, deviceDst, sizeof(float) * totalPixels, cudaMemcpyDeviceToHost);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
    cudaFree(deviceMsk);
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
    
    int numRows = (int)mxGetM(prhs[0]);
    int numCols = (int)mxGetN(prhs[0]);
    int numKRows = (int)mxGetM(prhs[1]);
    int numKCols = (int)mxGetN(prhs[1]);

    if (numKRows != 3 || numKCols != 3)
        mexErrMsgTxt("Invalid kernel size. It must be 3x3");
    
    plhs[0] = mxCreateNumericMatrix(numRows, numCols, mxSINGLE_CLASS, mxREAL);
    float* out = (float*)mxGetData(plhs[0]);

    conv2Mex(image, out, numRows, numCols, kernel);
}
