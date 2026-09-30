function [BestF,BestX,HisBestF]=FDBARO(nPop,MaxIt,Low,Up,Dim,problem)

% [nPop,Dim,maxFEs,Low,Up] = problem_terminate();

% MaxIt=ceil(maxFEs/nPop);

PopPos=zeros(nPop,Dim);
PopFit=zeros(nPop,1);

for i=1:nPop
    PopPos(i,:)=rand(1,Dim).*(Up-Low)+Low;
    PopFit(i)=problem(PopPos(i,:));
    
end

BestF=inf;
BestX=[];

for i=1:nPop
    if PopFit(i)<=BestF
        BestF=PopFit(i);
        BestX=PopPos(i,:);
    end
end

for It=1:MaxIt
    Direct1=zeros(nPop,Dim);
    Direct2=zeros(nPop,Dim);
    theta=2*(1-It/MaxIt);
    for i=1:nPop
        L=(exp(1)-exp(((It-1)/MaxIt)^2))*(sin(2*pi*rand)); %Eq.(3)
        rd=ceil(rand*(Dim));
        Direct1(i,randperm(Dim,rd))=1;
        c=Direct1(i,:); %Eq.(4)
        R=L.*c; %Eq.(2)
        
        A=2*log(1/rand)*theta;%Eq.(15)

        if A>1
            K=[1:i-1 i+1:nPop];
            RandInd=K(randi([1 nPop-1]));
%% Apply dynamic fitness-distance-balance (dFDB) method         
           
         fdbIndex= dFDB(PopPos,PopFit,MaxIt,It);
          
%% Update solution candidate using guide obtained by dFDB method             
            newPopPos=PopPos(RandInd,:)+R.*( PopPos(i,:)-PopPos(fdbIndex,:))...
                +round(0.5*(0.05+rand))*randn; %Eq.(1)
            
        else

            Direct2(i,ceil(rand*Dim))=1;
            gr=Direct2(i,:); %Eq.(12)
            H=((MaxIt-It+1)/MaxIt)*randn; %Eq.(8)
            b=PopPos(i,:)+H*gr.*PopPos(i,:); %Eq.(13)
            newPopPos=PopPos(i,:)+ R.*(rand*b-PopPos(i,:)); %Eq.(11)

        end
        newPopPos=SpaceBound(newPopPos,Up,Low);        
        newPopFit=problem(newPopPos);
                
        if newPopFit<PopFit(i)
            PopFit(i)=newPopFit;
            PopPos(i,:)=newPopPos;
        end
    end

    for i=1:nPop
        if PopFit(i)<BestF
            BestF=PopFit(i);
            BestX=PopPos(i,:);
        end
    end
    
    HisBestF(It)=BestF;
end


    
end


function  X=SpaceBound(X,Up,Low)

    Dim=length(X);
    S=(X>Up)+(X<Low);    
    X=(rand(1,Dim).*(Up-Low)+Low).*S+X.*(~S);

end
