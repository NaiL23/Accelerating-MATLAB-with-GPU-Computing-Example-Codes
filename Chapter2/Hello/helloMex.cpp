#include "mex.h"

//To compile: mex helloMex.cpp

void mexFunction(int nlhs, mxArray *plhs, int nrhs, mxArray *prhs) {
    mexPrintf("Hello, mex!\n");
}