function [Best_rime_rate,Best_rime,Convergence_curve]=TERIME(N,MaxFEs,lb,ub,dim,fobj)
% disp('RIME is now tackling your problem')
% initialize position

Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems

Rimepop=initialization(N,dim,ub,lb);%Initialize the set of random solutions
Lb=lb.*ones(1,dim);% lower boundary
Ub=ub.*ones(1,dim);% upper boundary
FEs=0;%Number of iterations
Convergence_curve=zeros(1,MaxFEs);
Rime_rates=zeros(1,N);%Initialize the fitness value
newRime_rates=zeros(1,N);
W = 5;%Soft-rime parameters, discussed in subsection 4.3.1 of the paper
Wmax=0.3;
Wmin=0.0625;
%Calculate the fitness value of the initial position
for i=1:N
         Rime_rates(1,i)=fobj(Rimepop(i,:));%Calculate the fitness value for each search agent
    %Make greedy selections
    if Rime_rates(1,i)<Best_rime_rate
        Best_rime_rate=Rime_rates(1,i);
        Best_rime=Rimepop(i,:);
    end
    
                     FEs=FEs+1;  
       Convergence_curve(FEs)=Best_rime_rate; 
         if FEs>MaxFEs
              break;
          end
end
% Main loop

while FEs <= MaxFEs
    %     it
    RimeFactor = (rand-0.5)*2*cos((pi*FEs/(MaxFEs*10)))*(1-round(FEs*W/MaxFEs)/W);%Parameters of Eq.(3),(4),(5)
    E =sqrt(FEs/MaxFEs);%Eq.(6)
    newRimepop = Rimepop;%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    if rand>0.5
        for i=1:N
            for j=1:dim
                %Soft-rime search strategy
                r1=rand();
                if r1< E
                    newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)
                end
            end
        end
    else
        for i=1:N
            newRimepop(i,:)=newRimepop(i,:)+rand*(newRimepop(randperm(N,1),:)-newRimepop(randperm(N,1),:));
        end
    end
    for i=1:N
        for j=1:dim
            %Hard-rime puncture mechanism
            r2=rand();
            if r2<normalized_rime_rates(i)
                r3=rand();
                if r3<0.5
                    C=(cos(pi*(FEs/MaxFEs))+1)*(1-FEs/MaxFEs)/2;
                    newRimepop(i,j)=newRimepop(i,j)+C*(newRimepop(randi(N),j)-newRimepop(randi(N),j));
                else
                    newRimepop(i,j)=normrnd(Best_rime(1,j),Best_rime(1,j)*0.001);
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
                     FEs=FEs+1;  
       Convergence_curve(FEs)=Best_rime_rate; 
         if FEs>MaxFEs
              break;
          end
    end

%     Convergence_curve(FEs)=Best_rime_rate;
%     FEs=FEs+1;
end
