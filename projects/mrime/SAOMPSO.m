% An adaptive snow ablation-inspired particle swarm optimization
function [Best_score,gBest,Convergence_curve]=SAOMPSO(N,FesMax,lb,ub,dim,fobj)

% Initializations
vel=zeros(N,dim);
pBestScore=zeros(N);
pBest=zeros(N,dim);
gBest=zeros(1,dim);
Convergence_curve=zeros(1,FesMax);
N1=floor(N*0.5);
Elite_pool=[];

for i=1:N
    pBestScore(i)=inf;
end

% Initialize gBestScore for a minimization problem
 gBestScore=inf;
% Random initialization for agents.
X=initialization(N,dim,ub,lb);
l=1;  
% Calculate the fitness of the first set and find the best one
for i=1:N
    fitness(i)=fobj(X(i,:));
    if(pBestScore(i)>fitness(i))
            pBestScore(i)=fitness(i);
            pBest(i,:)=X(i,:);
        end
        if(gBestScore>fitness(i))
            gBestScore=fitness(i);
            gBest=X(i,:);
        end 
                Convergence_curve(l)=gBestScore;
          l=l+1;
          if l>FesMax
              break;
          end    
        
end

[~,idx1]=sort(fitness);
second_best=X(idx1(2),:);
third_best=X(idx1(3),:);
sum1=0;

for i=1:N1
    sum1=sum1+X(idx1(i),:);
end

half_best_mean=sum1/N1;
Elite_pool(1,:)=gBest;
Elite_pool(2,:)=second_best;
Elite_pool(3,:)=third_best;
Elite_pool(4,:)=half_best_mean;

for i=1:N
    index(i)=i;
end

Na=ceil(N/2);
Nb=N-Na;
   
%Main loop
while l<=FesMax
    RB=randn(N,dim); %Brownian random number vector
    T=exp(-l/FesMax);
    k=1;
    DDF=0.35*(1+(5/7)*(exp(l/FesMax)-1)^k/(exp(1)-1)^k);
    M=DDF*T;
    %% Calculate the centroid position of the entire population
    for j=1:dim
        sum1=0;
        for i=1:N
            sum1=sum1+X(i,j);
        end
        X_centroid(j)=sum1/N;
    end
    
    %% Select individuals randomly to construct pop1 and pop2
    index1=randperm(N,Na);
    index2=setdiff(index,index1); 

    %Update the W of PSO
     w=0.3*sin((2*pi*l)/FesMax)+0.6;
     c1=sin((3*pi*l)/FesMax)+1.5;
     c2=cos((3*pi*l)/FesMax)+1.5;
     c=c1+c2;
     f=2/(2+log10(c)+abs(1-sqrt(c)/2));
    %% Exploration Phase（pop1）
    for i=1:Na 
        for j=1:size(X,2)
            Vmax=6*(1-log10(1+9*l/1000));
            vel(index1(i),j)=RB(index1(i),j)*w*vel(index1(i),j)+f*rand()*((pBest(index1(i),j)-X(index1(i),j))+(gBest(j)-X(index1(i),j))+(gBest(j)-pBest(index1(i),j)));
            if(vel(index1(i),j)>Vmax)
                vel(index1(i),j)=Vmax;
            end
            if(vel(index1(i),j)<-Vmax)
                vel(index1(i),j)=-Vmax;
            end            
            X(index1(i),j)=X(index1(i),j)+vel(index1(i),j);
            X(index1(i),j) = max(X(index1(i),j),lb(j));
            X(index1(i),j) = min(X(index1(i),j),ub(j));
        end
    end
    
    if Na<N
    Na=Na+1;
    Nb=Nb-1;
    end 

    %% Exploration of Elite Pool Leaders（pop2）
    if Nb>=1
    for i=1:Nb
        r2=2*rand-1;
         k1=randperm(4,1);
        for j=1:size(X,2)
            X(index2(i),j)=Elite_pool(k1,j)+RB(index2(i),j)*(r2*(gBest(j)-X(index2(i),j))+(1-r2)*(X_centroid(j)-X(index2(i),j)));
            X(index2(i),j) = max(X(index2(i),j),lb(j));
            X(index2(i),j) = min(X(index2(i),j),ub(j));
        end
    end
    end

    for i = 1:size(X,1)  
           fitness(i)=fobj(X(i,:));
        if(pBestScore(i)>fitness(i))
            pBestScore(i)=fitness(i);
            pBest(i,:)=X(i,:);
        end
        if(gBestScore>fitness(i))
            gBestScore=fitness(i);
            gBest=X(i,:);
        end
                        Convergence_curve(l)=gBestScore;
          l=l+1;
          if l>FesMax
              break;
          end  
        
    end 

    %% Exploration and exploitation of the entire population after pop2 is cleared to zero（Na=N）    
    w=0.3*sin((2*pi*l)/FesMax)+0.6;
    if Nb==0
    for i=1:Na 
        r1=rand;
        k1=randperm(4,1);
        for j=1:size(X,2)
            Vmax=6*(1-log10(1+9*l/1000));
            if rand<0.5
                X(i,j)=Elite_pool(k1,j)+RB(i,j)*(r1*(gBest(j)-X(i,j))+(1-r1)*(X_centroid(j)-X(i,j)));  %探索      
            else
                vel(i,j)=w*vel(i,j)+M*rand()*((pBest(i,j)-X(i,j))+(gBest(j)-X(i,j))+(gBest(j)-pBest(i,j))); %开发
            end
            if(vel(i,j)>Vmax)
                vel(i,j)=Vmax;
            end
            if(vel(i,j)<-Vmax)
                vel(i,j)=-Vmax;
            end            
            X(i,j)=X(i,j)+vel(i,j);
            X(i,j) = max(X(i,j),lb(j));
            X(i,j) = min(X(i,j),ub(j));
        end
    end
    end

    for i = 1:size(X,1)  
       fitness(i)=fobj(X(i,:));
        if(pBestScore(i)>fitness(i))
            pBestScore(i)=fitness(i);
            pBest(i,:)=X(i,:);
        end
        if(gBestScore>fitness(i))
            gBestScore=fitness(i);
            gBest=X(i,:);
        end
        
                        Convergence_curve(l)=gBestScore;
%           l=l+1;
%           if l>FesMax
%               break;
%           end  
        
    end

%% Update the elite pool
    [~,idx1]=sort(fitness);
    second_best=X(idx1(2),:);
    third_best=X(idx1(3),:);
    sum1=0;
    for i=1:N1
        sum1=sum1+X(idx1(i),:);
    end
    half_best_mean=sum1/N1;
    Elite_pool(1,:)=gBest;
    Elite_pool(2,:)=second_best;
    Elite_pool(3,:)=third_best;
    Elite_pool(4,:)=half_best_mean;

%     Convergence_curve(l)=gBestScore;
%     l=l+1;
end
Best_score=min(Convergence_curve);
% time=toc;
end
