function Positions=Cubicinitialization(SearchAgents_no,dim,ub,lb)

Cubicsequence=Cubic();
lenn=length(Cubicsequence);

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    for i=1:SearchAgents_no
        for j=1:dim
            chars(i,j)=Cubicsequence(rem(i+j,lenn));
        end
    end
    Positions=chars.*(ub-lb)+lb;
    %rand(SearchAgents_no,dim)
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        for i=1:SearchAgents_no
            chars(1,i)=Cubicsequence(rem(i,lenn));
        end
        Positions(i,:)=chars.*(ub_i-lb_i)+lb_i;
        %rand(SearchAgents_no,1)
    end
end