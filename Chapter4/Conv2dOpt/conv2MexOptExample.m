quarters = single(imread('eight.tif'));
kernel = single([1 2 1; 0 0 0; -1 -2 -1]);

figure(1)
imagesc(quarters);
colormap(gray);

% mexcuda conv2MexNaive.cu
tic
H1 = conv2MexNaive(quarters, kernel);
toc
figure(2)
imagesc(H1);
colormap(gray);

% mexcuda conv2MexOptA.cu
tic
H2 = conv2MexOptA(quarters, kernel);
toc
figure(3)
imagesc(H2);
colormap(gray);

% mexcuda conv2MexOptB.cu
tic
H3 = conv2MexOptB(quarters, kernel);
toc
figure(4)
imagesc(H3);
colormap(gray);
