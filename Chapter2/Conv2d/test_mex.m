quarters = single(imread('eight.tif'));
kernel = single([1 2 1; 0 0 0; -1 -2 -1]);

figure(1)
imagesc(quarters);
colormap(gray);

tic
H = conv2Mex(quarters, kernel);
toc

figure(3)
imagesc(H);
colormap(gray);