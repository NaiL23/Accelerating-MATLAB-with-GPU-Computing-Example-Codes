% generate sample volume data
[X, Y, Z, V] = flow;
X = single(X);
Y = single(Y);
Z = single(Z);
V = single(V);
isovalue = single(-3);

tic
[Vertices2, Indices2] = getSurfaceWithOpt(X, Y, Z, V, isovalue);
toc

% visualize triangles
figure
p = patch('Faces', Indices2, 'Vertices', Vertices2);
set(p, 'FaceColor', 'none', 'EdgeColor', 'green');
daspect([1,1,1])
view(3);
camlight
lighting gouraud
grid on
