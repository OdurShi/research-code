function [NewX , Sol_CostNew] = Boundary_Check(NewX,fhd,varargin,LB,UB)
    for j=1:4
        NewX(j,:) = boundaryCheck(NewX(j,:), LB, UB);
%         Sol_CostNew(j,:)=fobj(NewX(j,:));%#ok
        Sol_CostNew(j,:)=feval(fhd, NewX(j,:)',varargin);
    end
            
end