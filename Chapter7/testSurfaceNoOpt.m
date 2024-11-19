% generate sample volume data
[X, Y, Z, V] = flow;

% visualize volume data
figure
xmin = min(X(:)); 
ymin = min(Y(:)); 
zmin = min(Z(:));
xmax = max(X(:)); 
ymax = max(Y(:)); 
zmax = max(Z(:));
hslice = surf(linspace(xmin,xmax,100), linspace(ymin,ymax,100), zeros(100));
rotate(hslice,[-1,0,0],-45)
xd = get(hslice,'XData');
yd = get(hslice,'YData');
zd = get(hslice,'ZData');
delete(hslice)
h = slice(X,Y,Z,V,xd,yd,zd);
set(h,'FaceColor', 'interp', 'EdgeColor', 'none', 'DiffuseStrength', 0.8)
hold on
hx = slice(X,Y,Z,V,xmax,[],[]);
set(hx,'FaceColor', 'interp', 'EdgeColor', 'none')
hy = slice(X,Y,Z,V,[],ymax,[]);
set(hy,'FaceColor', 'interp', 'EdgeColor', 'none')
hz = slice(X,Y,Z,V,[],[],zmin);
set(hz,'FaceColor', 'interp', 'EdgeColor', 'none')

% data type to single
X = single(X);
Y = single(Y);
Z = single(Z);
V = single(V);
isovalue = single(-3);

% Marching cubes
tic
[Vertices1, Indices1] = getSurfaceNoOpt(X, Y, Z, V, isovalue);
toc

% visualize triangles
figure
p = patch('Faces', Indices1, 'Vertices', Vertices1);
set(p, 'FaceColor', 'none', 'EdgeColor', 'red');
daspect([1,1,1])
view(3);
camlight
lighting gouraud
grid on
