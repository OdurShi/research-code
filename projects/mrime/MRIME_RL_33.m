
function [Best_rime_rate,Best_rime,Convergence_curve]=MRIME_RL_33(N,FesMax,lb,ub,dim,fobj)

% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems
X=initialization(N,dim,ub,lb);%Initialize the set of random solutions
Lb=lb.*ones(1,dim);% lower boundary 
Ub=ub.*ones(1,dim);% upper boundary

lu = [Lb; Ub];
I=zeros(1,N);
it=1;%Number of iterations
Convergence_curve=zeros(1,FesMax);
Rime_rates=zeros(1,N);%Initialize the fitness value
newRime_rates=zeros(1,N);
W = 5;%Soft-rime parameters, discussed in subsection 4.3.1 of the paper
%Calculate the fitness value of the initial position
for i=1:N
    Rime_rates(1,i)=fobj(X(i,:));%Calculate the fitness value for each search agent
    %Make greedy selections
    if Rime_rates(1,i)<Best_rime_rate
        Best_rime_rate=Rime_rates(1,i);
        Best_rime=X(i,:);
    end
end
% Main loop
while it <= FesMax
    
            Par_restart.lu = lu;
        Par_restart.V_lim = 1;
        for i=1:dim
            Par_restart.V_lim = (Par_restart.V_lim .*(abs(Par_restart.lu(1,i) - Par_restart.lu(2,i))));
        end
        Par_restart.V_lim = sqrt(Par_restart.V_lim);
        Par_restart.counter = zeros(N,1);
        Par_restart.m = 1;
        Par_restart.m_max = 4;
        Par_restart.m_min = 1;
    
    
    RimeFactor = (rand-0.5)*2*cos((pi*it/(FesMax/10)))*(1-round(it*W/FesMax)/W);%Parameters of Eq.(3),(4),(5)
    E =(it/FesMax)^0.5;%Eq.(6)
    newRimepop = X;%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    
    
          [temp_fit, sorted_index] = sort(Rime_rates, 'ascend');
                fdbIndex =RFDB(X, Rime_rates);
     %%%%% Choose neighbourhood region to the best individual
                best = X(fdbIndex, :);
                Dis = pdist2(X,best,'euclidean'); % euclidean distance
                %D2 = sqrt(sum((pop(1,:) - best).^2, 2));

                %%%% Sort
                [Dis_ordered idx_ordered] = sort(Dis, 'ascend');
                 SEL=round(0.5*N);
                Neighbour_best_pool = X(idx_ordered(1:SEL), :); %%% including best also so start from 1
                Xsel = Neighbour_best_pool;
                                                    weights = log(SEL+1/2)-log(1:SEL)';
weights = weights/sum(weights);  
    xmean = weights'*Xsel;    % Eq.10
%                 xmean = mean(Xsel);
                % covariance matrix calculation
                C =  1/(SEL-1)*(Xsel - xmean(ones(SEL,1), :))'*(Xsel - xmean(ones(SEL,1), :));
                C = triu(C) + transpose(triu(C,1)); % enforce symmetry
                [R,D] = eig(C);
                    if max(diag(D)) > 1e20*min(diag(D))

                    tmp = max(diag(D))/1e20 - min(diag(D));
                    C = C + tmp*eye(dim);
                    [R, D] = eig(C);
                    end
                    D  = sqrt(diag(D+0.0000001*eye(dim)));
              zz=randn(N,dim);    
    
    
    for i=1:N
        for j=1:dim
            %Soft-rime search strategy
            r1=rand();
            if r1< E
                newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)
                
            end
            %Hard-rime puncture mechanism
            r2=rand();
            if r2<normalized_rime_rates(i)
                newRimepop(i,j)=Best_rime(1,j);%Eq.(7)
            end
        end
    end
    for i=1:N
        %Boundary absorption
        Flag4ub=newRimepop(i,:)>ub;
        Flag4lb=newRimepop(i,:)<lb;
        newRimepop(i,:)=(newRimepop(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        newRime_rates(1,i)=fobj(newRimepop(i,:));
        %Positive greedy selection mechanism
        if newRime_rates(1,i)<Rime_rates(1,i)
            Rime_rates(1,i) = newRime_rates(1,i);
            X(i,:) = newRimepop(i,:);
            if newRime_rates(1,i)< Best_rime_rate
               Best_rime_rate=Rime_rates(1,i);
               Best_rime=X(i,:);
            end
            
        else
            I(i)=I(i)+1;
        end
               it=it+1;  
       Convergence_curve(it)=Best_rime_rate; 
         if it>FesMax
              break;
          end
    end
    
%        [X,Rime_rates,Par_restart]=Restart_mechanism(X,Rime_rates,Par_restart,I,Best_rime_rate,1,it);
       
[~, bes_l]=min(Rime_rates);
[PopSize,n]=size(X);
V_pop = 1;
for j=1:n
    V_pop = V_pop.*abs((max(X(:,j)-min(X(:,j)))))./2;
end
V_pop = sqrt(V_pop);
nVOL = sqrt(V_pop/Par_restart.V_lim);
if nVOL < 0.001
    for i = 1:PopSize
            if  I(i)>2*n && i~=bes_l
AAAAA=randperm(N);
       randnum=rand(1,3*dim);
     RSRimepop=xmean+randnum(1)*(X(AAAAA(2),:)-X(i,:))+randnum(2)*(X(AAAAA(1),:)-X(i,:))+(R*(D.*(zz(i,:)')))';  
      

       %Boundary absorption
        Flag4ub=RSRimepop>ub;
        Flag4lb=RSRimepop<lb;
        RSRimepop=(RSRimepop.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        RSRime_rates=fobj(RSRimepop);
        %Positive greedy selection mechanism
        if RSRime_rates<Rime_rates(1,i)
            Rime_rates(1,i) = RSRime_rates;
            X(i,:) = RSRimepop;
              I(i)=0;
            if RSRime_rates< Best_rime_rate
               Best_rime_rate=Rime_rates(1,i);
               Best_rime=X(i,:);
            end
            
        else
            I(i)=I(i)+1;
        end
               it=it+1;  
       Convergence_curve(it)=Best_rime_rate; 
         if it>FesMax
              break;
          end
  
            end
  
    end
end



end



