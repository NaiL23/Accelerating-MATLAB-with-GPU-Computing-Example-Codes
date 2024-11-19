% mexcuda cublasDemo.cu -lcublas
A = single(rand(200, 300));
B = single(rand(300, 400));
C = cublasDemo(A, B);
