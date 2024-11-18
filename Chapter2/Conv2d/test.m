quarters = single(imread('eight.tif'));
kernel = single([1 2 1; 0 0 0; -1 -2 -1]);

figure(1)
imagesc(quarters);
colormap(gray);

tic
H1 = conv2(quarters, kernel, 'same');
toc
figure(2)
imagesc(H1);
colormap(gray);

% mex conv2Mex.cpp
tic
H2 = conv2Mex(quarters, kernel);
toc
figure(3)
imagesc(H2);
colormap(gray);

% mexcuda conv2Cuda.cu
tic
H3 = conv2Cuda(quarters, kernel);
toc
figure(4)
imagesc(H3);
colormap(gray);