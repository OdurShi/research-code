%--------------------------------------------------------------------------
% MRFO code v1.0.
% Developed in MATLAB R2011b
% The code is based on the following papers.
% W. Zhao, Z. Zhang and L. Wang, Manta ray foraging optimization: An
% effective bio-inspired optimizer for engineering applications.
% Engineering Applications of Artifical Intelligence (2019),
% https://dio.org/10.1016/j.engappai.2019.103300.
% -------------------------------------------------------------------------
% FunIndex: Index of function.
% MaxIt: The maximum number of iterations.
% PopSize: The size of population.
% PopPos: The position of population.
% PopFit: The fitness of population.
% Dim: The dimensionality of prloblem.
% Alpha: The weight coefficient in chain foraging.
% Beta: The weight coefficient in cyclone foraging.
% S: The somersault factor.
% BestF: The best fitness corresponding to BestX. 
% HisBestFit: History best fitness over iterations. 
% Low: The low bound of search space.
% Up: The up bound of search space.

function [BestF,BestX,HisBestFit]=MRFO(nPop,MaxIt,Low,Up,Dim,fobj)
 

    for i=1:nPop   
        PopPos(i,:)=rand(1,Dim).*(Up-Low)+Low;
    PopFit(i)=fobj(PopPos(i,:)); 
    end
       BestF=inf;
       BestX=[];

    for i=1:nPop
        if PopFit(i)<=BestF
           BestF=PopFit(i);
           BestX=PopPos(i,:);
        end
    end

       HisBestFit=zeros(MaxIt,1);

It=1;
while It<MaxIt+1  
     Coef=It/MaxIt; 
     
       if rand<0.5
          r1=rand;                         
          Beta=2*exp(r1*((MaxIt-It+1)/MaxIt))*(sin(2*pi*r1));    
          if  Coef>rand                                                      
              newPopPos(1,:)=BestX+rand(1,Dim).*(BestX-PopPos(1,:))+Beta*(BestX-PopPos(1,:)); %Equation (4)
          else
              IndivRand=rand(1,Dim).*(Up-Low)+Low;                                
              newPopPos(1,:)=IndivRand+rand(1,Dim).*(IndivRand-PopPos(1,:))+Beta*(IndivRand-PopPos(1,:)); %Equation (7)         
          end              
       else 
            Alpha=2*rand(1,Dim).*(-log(rand(1,Dim))).^0.5;           
            newPopPos(1,:)=PopPos(1,:)+rand(1,Dim).*(BestX-PopPos(1,:))+Alpha.*(BestX-PopPos(1,:)); %Equation (1)
       end
     
    for i=2:nPop
        if Coef>rand
           r1=rand;                         
           Beta=2*exp(r1*((MaxIt-It+1)/MaxIt))*(sin(2*pi*r1));    
             if  Coef>rand                                                      
                 newPopPos(i,:)=BestX+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Beta*(BestX-PopPos(i,:)); %Equation (4)
             else
                 IndivRand=rand(1,Dim).*(Up-Low)+Low;                                
                 newPopPos(i,:)=IndivRand+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Beta*(IndivRand-PopPos(i,:));  %Equation (7)       
             end              
        else
            Alpha=2*rand(1,Dim).*(-log(rand(1,Dim))).^0.5;           
            newPopPos(i,:)=PopPos(i,:)+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Alpha.*(BestX-PopPos(i,:)); %Equation (1)
       end         
    end
         
           for i=1:nPop        
               newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
             newPopFit(i)=fobj(newPopPos(i,:));      
              if newPopFit(i)<PopFit(i)
                 PopFit(i)=newPopFit(i);
                 PopPos(i,:)=newPopPos(i,:);
              end
           end
     
            S=2;
          %  S=2*(1-It/MaxIt)^2;
        for i=1:nPop           
            newPopPos(i,:)=PopPos(i,:)+S*(rand*BestX-rand*PopPos(i,:)); %Equation (8)
        end
     
     for i=1:nPop        
         newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
         newPopFit(i)=fobj(newPopPos(i,:));   
         if newPopFit(i)<PopFit(i)
            PopFit(i)=newPopFit(i);
            PopPos(i,:)=newPopPos(i,:);
         end
     end
     
     for i=1:nPop
        if PopFit(i)<BestF
           BestF=PopFit(i);
           BestX=PopPos(i,:);            
        end
             HisBestFit(It)=BestF;
             It=It+1;
                           if It>MaxIt
              break;
          end
              HisBestFit(It)=BestF;
             It=It+1;
              if It>MaxIt
              break;
          end
        
        
        
     end
 
  

end


function  X=SpaceBound(X,Up,Low)


    Dim=length(X);
    S=(X>Up)+(X<Low);    
    X=(rand(1,Dim).*(Up-Low)+Low).*S+X.*(~S);

end
end
