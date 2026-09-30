function [popu] = BoundaryDetection1(popu,Xmin,Xmax)
[NP, D] = size(popu);
xl = repmat(Xmin, NP, D);
xu = repmat(Xmax, NP, D);
ranX=rand(NP, D);
%check the lower bound
pos1 = popu < xl;
popu(pos1) = xl(pos1) + ranX(pos1) .* (xu(pos1) - xl(pos1));
%check the upper bound
pos2 = popu > xu;
popu(pos2) = xl(pos2) + ranX(pos2) .* (xu(pos2) - xl(pos2));
end