% generate sample volume data
[X, Y, Z, V] = flow;
X = single(X);
Y = single(Y);
Z = single(Z);
V = single(V);
isovalue = single(-3);

tic
[Vertices3, Indices3] = getSurfaceCuda(X, Y, Z, V, isovalue);
toc

% visualize triangles
figure
p = patch('Faces', Indices3, 'Vertices', Vertices3);
set(p, 'FaceColor', 'none', 'EdgeColor', 'blue');
daspect([1,1,1])
view(3);
camlight
lighting gouraud
grid on
