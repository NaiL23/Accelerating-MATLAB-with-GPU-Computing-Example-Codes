#include "mex.h"
// To compile: mexcuda conv2MexOptB.cu

__global__ void conv2MexCuda(float* src,
                             float* dst,
                             int numRows,
                             int numCols,
                             float* mask)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 1 || row > numRows - 1 || col < 1 || col > numCols - 1)
        return;

    __shared__ float sharedMask[9];
    if (threadIdx.x < 9)
    {
        sharedMask[threadIdx.x] = mask[threadIdx.x];
    }
    __syncthreads();
    

    int dstIndex = col * numRows + row;
    dst[dstIndex] = 0;
    int mskIndex = 8;
    for (int kc = -1; kc < 2; kc++)
    {
        int srcIndex = (col + kc) * numRows + row;
        for (int kr = -1; kr < 2; kr++)
        {
            dst[dstIndex] += sharedMask[mskIndex--] * src[srcIndex + kr];
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

    const int size = 16;
    dim3 blockSize(size, size);
    dim3 gridSize((numRows + size - 1) / blockSize.x,
                  (numCols + size - 1) / blockSize.y);

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

