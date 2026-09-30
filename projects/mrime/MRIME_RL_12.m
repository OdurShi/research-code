
function [Best_rime_rate,Best_rime,Convergence_curve]=MRIME_RL_12(N,FesMax,lb,ub,dim,fobj)

% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems
X=initialization(N,dim,ub,lb);%Initialize the set of random solutions
Lb=lb.*ones(1,dim);% lower boundary 
Ub=ub.*ones(1,dim);% upper boundary

Cov_Store_Pos=[];
Cov_Store_Fit=[];
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
    RimeFactor = (rand-0.5)*2*cos((pi*it/(FesMax/10)))*(1-round(it*W/FesMax)/W);%Parameters of Eq.(3),(4),(5)
    E =(it/FesMax)^0.5;%Eq.(6)
    newRimepop = X;%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    
       [~, index]   = sort(Rime_rates);
    Cov_Store_Pos= [Cov_Store_Pos;X(index(1:round(N*0.3)),:)];
    Cov_Store_Fit= [Cov_Store_Fit Rime_rates(index(1:round(N*0.3)))];

    Cov_Store_Pos_Num=size(Cov_Store_Pos,1);
    if Cov_Store_Pos_Num>25*dim
        NUm=Cov_Store_Pos_Num-25*dim;
        Cov_Store_Pos(1:NUm,:)=[];
        Cov_Store_Fit(1:NUm)=[];
    end
    
    
    %     if 25*dim>floor(N*0.5)
         if 25*dim>N
          [temp_fit, sorted_index] = sort(Cov_Store_Fit, 'ascend');
                fdbIndex =RFDB(Cov_Store_Pos, Cov_Store_Fit);
     %%%%% Choose neighbourhood region to the best individual
                best = Cov_Store_Pos(fdbIndex, :);
                Dis = pdist2(Cov_Store_Pos,best,'euclidean'); % euclidean distance
                %%%% Sort
                [Dis_ordered idx_ordered] = sort(Dis, 'ascend');
              SEL=floor(Cov_Store_Pos_Num*0.5);
                Neighbour_best_pool = Cov_Store_Pos(idx_ordered(1:SEL), :); %%% including best also so start from 1
    else
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
    end

            
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
% 
   
    for i=1:N
              r4=rand;
             r1=rand();
      if r1<E
        for j=1:dim
            %Soft-rime search strategy
%             r1=rand();
                       r3=rand();
            if r3< E
                newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)                
            end
            %Hard-rime puncture mechanism
            r2=rand();
%                r4=rand;
            if r2<normalized_rime_rates(i)
                               if r4< E
                newRimepop(i,j)=best(1,j);%Eq.(7)
                else
                 newRimepop(i,j)=(best(1,j)+xmean(1,j))/2;%Eq.(7)   
                end
            end
        end
    else
              randnum=rand(1,3*dim);
     newRimepop(i,:)=xmean+randnum(1)*(xmean-X(i,:))+(R*(D.*(zz(i,:)')))';  
             for j=1:dim
            %Soft-rime search strategy
            %Hard-rime puncture mechanism
            r1=rand();
            r2=rand();
            if r2<normalized_rime_rates(i)
                               if r4< E
                newRimepop(i,j)=best(1,j);%Eq.(7)
                else
                 newRimepop(i,j)=(best(1,j)+xmean(1,j))/2;%Eq.(7)   
                end
            end
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
        end
               it=it+1;  
       Convergence_curve(it)=Best_rime_rate; 
         if it>FesMax
              break;
          end
    end

end



