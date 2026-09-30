
function [Best_rime_rate,Best_rime,Convergence_curve]=RIME_RL(N,FesMax,lb,ub,dim,fobj)

% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems
X=initialization(N,dim,ub,lb);%Initialize the set of random solutions
Lb=lb.*ones(1,dim);% lower boundary 
Ub=ub.*ones(1,dim);% upper boundary
it=1;%Number of iterations
Convergence_curve=zeros(1,FesMax);
Rime_rates=zeros(1,N);%Initialize the fitness value
newRime_rates=zeros(1,N);
W = 5;%Soft-rime parameters, discussed in subsection 4.3.1 of the paper


counter=zeros(1,N);
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
            counter(i)=counter(i)+1;
        end
               it=it+1;  
       Convergence_curve(it)=Best_rime_rate; 
         if it>FesMax
              break;
          end
    end
    
    for i=1:N
     if     counter(i)>2*dim
                        nDim_j = randi(dim);
                nSeq_j=randperm(dim);
                j=nSeq_j(1:nDim_j);
                [~,j_size] = size(j);
                
                 nDim_i = randi(N);
                nSeq_i1=randperm(N);
                j_i1=nSeq_i1(1:nDim_i);
                nSeq_i2=randperm(N);
                j_i2=nSeq_i2(1:nDim_i);
                [~,j_size_i1] = size(j_i1);
                  if rand > 0.5
                    for i1=1:j_size_i1
                        newRimepop(i,:) = rand * X(j_i1(i1),:) + (1-rand)*(X(j_i2(i1),:)) + rands(1,1)*(X(j_i1(i1),:)-X(j_i2(i1),:));
                        %x(j_i2(i1),:) = rand * x(j_i2(i1),:) + (1-rand)*(x(j_i1(i1),:)) + rands(1,1)*(x(j_i2(i1),:)-x(j_i1(i1),:));
                    end
                else
                    for j1 = 1:j_size
                        j_num = j(1,j1);
                        pop_num = X(:,j_num);
                        pop_num_1 = pop_num(randperm(numel(pop_num),1));
                        pop_num_11 = X(i,j_num);
                        newRimepop(i,j_num) = rand * pop_num_11 + (1-rand)*pop_num_1;
                    end
                  end
                
                  
                  
                  
     end 
        
    end
    
    
    
    

end



