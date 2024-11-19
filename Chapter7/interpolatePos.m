function [varargout] = interpolatePos(isovalue, voxel1, voxel2)
    scale = (isovalue - voxel1(:,1)) ./ (voxel2(:,1) - voxel1(:,1));
    interpolatedPos = voxel1(:,2:4) + [scale .* (voxel2(:,2) - voxel1(:,2)), ...
                                       scale .* (voxel2(:,3) - voxel1(:,3)), ...
                                       scale .* (voxel2(:,4) - voxel1(:,4))];

    if nargout == 1 || nargout == 0
        varargout{1} = interpolatedPos;
    elseif nargout == 3
        varargout{1} = interpolatedPos(:,1);
        varargout{2} = interpolatedPos(:,2);
        varargout{3} = interpolatedPos(:,3);
    end
                                   