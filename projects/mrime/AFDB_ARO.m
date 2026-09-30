function [BestF,BestX,HisBestF]=AFDB_ARO(nPop,MaxIt,Low,Up,Dim,fobj)



BestF=inf;
BestX=[]; 
FE=0;

PopPos=zeros(nPop,Dim);
PopFit=zeros(nPop,1);

for i=1:nPop
    PopPos(i,:)=rand(1,Dim).*(Up-Low)+Low;
    PopFit(i)=fobj(PopPos(i,:));
end

for i=1:nPop
    if PopFit(i)<=BestF
        BestF=PopFit(i);
        BestX=PopPos(i,:);
    end
end
HisBestF=zeros(MaxIt,1);
It=1;
while It<MaxIt+1
    
    Direct1=zeros(nPop,Dim);
    Direct2=zeros(nPop,Dim);
    theta=2*(1-It/MaxIt);
       
    for i=1:nPop
          L=(exp(1)-exp(((It-1)/MaxIt)^2))*(sin(2*pi*rand)); %Eq.(3)
        rd=ceil(rand*(Dim));
        Direct1(i,randperm(Dim,rd))=1;
        c=Direct1(i,:); %Eq.(4)
        R=L.*c; %Eq.(2)
        
        %%% the energy factor is designed to model the switch from exploration to exploitation.
        A=2*log(1/rand)*theta;%Eq.(15)
       
        %%%Detour foraging (exploration)
        if A>1
            rand_num=rand;
            if rand_num < ActFun_Sin_SF_Decrease(MaxIt, It)
            fdbindex=fitnessDistanceBalance(PopPos, PopFit);
            RandInd_fdb=fdbindex;
            K=[1:i-1 i+1:nPop];  
            RandInd=K(randi([1 nPop-1])); 
            newPopPos=PopPos(RandInd_fdb,:)+R.*(PopPos(i,:)-PopPos(RandInd,:))+round(0.5*(0.05+rand))*randn; %Eq.(1)
            else
            K=[1:i-1 i+1:nPop];  
            RandInd=K(randi([1 nPop-1])); 
            newPopPos=PopPos(RandInd,:)+R.*(PopPos(i,:)-PopPos(RandInd,:))+round(0.5*(0.05+rand))*randn; %Eq.(1)
            end
        else
            
        %%%Random hiding (exploitation)
            Direct2(i,ceil(rand*Dim))=1;
            gr=Direct2(i,:); %Eq.(12)
            H=((MaxIt-It+1)/MaxIt)*randn; %Eq.(8)
            b=PopPos(i,:)+H*gr.*PopPos(i,:); %Eq.(13)
            newPopPos=PopPos(i,:)+ R.*(rand*b-PopPos(i,:)); %Eq.(11)
        end
        
        newPopPos=SpaceBound(newPopPos,Up,Low);
        newPopFit=fobj(newPopPos);
       
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
        
         HisBestF(It)=BestF;
        It=It+1;
        
           if It> MaxIt
              break;
          end
        
      
        
    end
      
end


  
end

function  X=SpaceBound(X,Up,Low)
    Dim=length(X);
    S=(X>Up)+(X<Low);    
    X=(rand(1,Dim).*(Up-Low)+Low).*S+X.*(~S);


end