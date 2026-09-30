
function Positions=initialization_YDSE(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

if Boundary_no==1
    Positions=Bernoulli(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=Bernoulli(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end