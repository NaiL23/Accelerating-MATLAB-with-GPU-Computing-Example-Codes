#include"mex.h"

//To compile: mexcuda AddVectors.cu

__global__ void addVectorsMask(float* A,float* B,float* C,int size)
{
    int i=blockIdx.x;
    if(i>=size)
    {
        return ;
    }
    C[i]=A[i]+B[i];
}
void addVectors(float* A,float* B,float* C,int size)
{
    float *devPtrA=0,*devPtrB=0,*devPtrC=0;
    cudaMalloc(&devPtrA,sizeof(float)*size);
    cudaMalloc(&devPtrB,sizeof(float)*size);
    cudaMalloc(&devPtrC,sizeof(float)*size);
    
    cudaMemcpy(devPtrA,A,sizeof(float)*size,cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB,B,sizeof(float)*size,cudaMemcpyHostToDevice);
    addVectorsMask<<<size,1>>>(devPtrA,devPtrB,devPtrC,size);
    cudaMemcpy(C,devPtrC,sizeof(float)*size,cudaMemcpyDeviceToHost);
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
}

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,mxArray* prhs[])
{
    if(nrhs!=2)
    {
      mexErrMsgTxt("Invaid number of input arguments");
    }
    if(nlhs!=1)
    {
        mexErrMsgTxt("Invaid number of outputs");
    }
    if(!mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1]))
    {
        mexErrMsgTxt("input vector data type must be single");
    }
    int numRowsA=(int)mxGetM(prhs[0]);
    int numColsA=(int)mxGetN(prhs[0]);
    int numRowsB=(int)mxGetM(prhs[1]);
    int numColsB=(int)mxGetN(prhs[1]);
    if(numRowsA!=numRowsB||numColsA!=numColsB)
    {
        mexErrMsgTxt("Invalid size.The sizes of two vectors must be same");
     }
    int minSize=(numRowsA<numColsA)?numRowsA:numColsA;
    int maxSize=(numRowsA>numColsA)?numRowsA:numColsA;
    if(minSize!=1)
    {
        mexErrMsgTxt("input vector data type must be one dimentional");
    }

    float* A=(float*)mxGetData(prhs[0]);
    float* B=(float*)mxGetData(prhs[1]);
    plhs[0] = mxCreateNumericMatrix(numRowsA,numColsB,mxSINGLE_CLASS,mxREAL);
    float* C=(float*)mxGetData(plhs[0]);
    
    addVectors(A,B,C,maxSize);
}