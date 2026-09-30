
function [Best_FF,Best_P,Conv_curve]=CSOAOA(N,FesMax,LB,UB,Dim,F_obj)

Best_P=zeros(1,Dim);
Best_FF=inf;
Conv_curve=zeros(1,FesMax);

%Initialize the positions of solution
X=initialization1(N,Dim,UB,LB);
Xnew=X;
Ffun=zeros(1,size(X,1));% (fitness values)
Ffun_new=zeros(1,size(Xnew,1));% (fitness values)

MOP_Max=1;
MOP_Min=0.2;
C_Iter=1;
Alpha=5;
Mu=0.499;
deta=0.001+0.09*rand;

for i=1:size(X,1)
    Ffun(1,i)=F_obj(X(i,:));  %Calculate the fitness values of solutions
    if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
    end
end
    
    
while C_Iter<FesMax+1  %Main loop
    MOP=1-((C_Iter)^(1/Alpha)/(FesMax)^(1/Alpha));   % Probability Ratio 
    MOA=MOP_Min+C_Iter*((MOP_Max-MOP_Min)/FesMax); %Accelerated function
   
    %Update the Position of solutions
    for i=1:size(X,1)   % if each of the UB and LB has a just value 
        for j=1:size(X,2)
           r1=rand();
            if (size(LB,2)==1)
                if r1>MOA
                    r2=rand();
                    if r2>0.5
                        Xnew(i,j)=(1+deta*rand)*Best_P(1,j)/(MOP+eps)*((UB-LB)*Mu+LB);
                    else
                        Xnew(i,j)=(1+deta*rand)*Best_P(1,j)*MOP*((UB-LB)*Mu+LB);
                    end
                else
                    r3=rand();
                    if r3>0.5
                        Xnew(i,j)=(1+deta*rand)*Best_P(1,j)-MOP*((UB-LB)*Mu+LB);
                    else
                        Xnew(i,j)=(1+deta*rand)*Best_P(1,j)+MOP*((UB-LB)*Mu+LB);
                    end
                end               
            end           
        end
        
        Flag_UB=Xnew(i,:)>UB; % check if they exceed (up) the boundaries
        Flag_LB=Xnew(i,:)<LB; % check if they exceed (down) the boundaries
        Xnew(i,:)=(Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
 
        Ffun_new(1,i)=F_obj(Xnew(i,:));  % calculate Fitness function 
        if Ffun_new(1,i)<Ffun(1,i)
            X(i,:)=Xnew(i,:);
            Ffun(1,i)=Ffun_new(1,i);
            
                if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
    end
            
        end 
        
        
                   
  Conv_curve(C_Iter)=Best_FF; 
  C_Iter=C_Iter+1;  
            if C_Iter>FesMax
              break;
          end
        
        
        
    end
    RandIndex = randperm(N);
    for i = 1:N/2
       r1 =rand;
       r2 = rand;
       c1 = 2*rand -1;
       c2 = 2*rand -1;
       j = ceil(N/2) + i;
       ii = RandIndex(i);
       jj = RandIndex(j);
       MSp=r1.*X(ii,:) + (1 - r1).*X(jj,:) + c1.*(X(ii,:)-X(jj,:));
       MSp1 = r2.*X(jj,:) + (1 - r2).*X(ii,:) + c2.*(X(jj,:)-X(ii,:));
       
       Flag_UB=MSp>UB; % check if they exceed (up) the boundaries
       Flag_LB=MSp<LB; % check if they exceed (down) the boundaries
       MSp=(MSp.*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
       Flag_UB=MSp1>UB; % check if they exceed (up) the boundaries
       Flag_LB=MSp1<LB; % check if they exceed (down) the boundaries
       MSp1=(MSp1.*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
       fit1 = F_obj(MSp);
       fit2 = F_obj(MSp1);
       if(fit1 <Ffun(1,ii))
          X(ii,:) = MSp;
          Ffun(1,ii) = fit1;
             if Ffun(1,ii)<Best_FF
        Best_FF=Ffun(1,ii);
        Best_P=X(ii,:);
    end
          
       end
         Conv_curve(C_Iter)=Best_FF; 
  C_Iter=C_Iter+1;  
            if C_Iter>FesMax
              break;
          end
       
       
       
       if(fit2<Ffun(1,jj))
          X(jj,:) = MSp1;
          Ffun(1,jj) = fit2;
                       if Ffun(1,jj)<Best_FF
        Best_FF=Ffun(1,jj);
        Best_P=X(jj,:);
    end
       end 
       
       
         Conv_curve(C_Iter)=Best_FF; 
  C_Iter=C_Iter+1;  
            if C_Iter>FesMax
              break;
          end
    end
    
    for i= 1:N
        RandIndexZ = randperm(Dim);
        index1 = RandIndexZ(1);
        index2 = RandIndexZ(2);
        r = rand;
        MSv = X(i,:);
        MSv(index1) = r.*X(i,index1) + (1 - r).*X(i,index2);
        
        Flag_UB=MSv>UB; % check if they exceed (up) the boundaries
        Flag_LB=MSv<LB; % check if they exceed (down) the boundaries
        MSv=(MSv.*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
        
        fitv = F_obj(MSv);
        if(fitv < Ffun(1,i))
            X(i,:) = MSv;
            Ffun(1,i) = fitv;
            
           if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
    end
            
        end 
        
               
         Conv_curve(C_Iter)=Best_FF; 
  C_Iter=C_Iter+1;  
            if C_Iter>FesMax
              break;
          end
        
        
        
    end
    
    ind = find(Ffun == min(Ffun), 1);                   
    Best_P = X(ind,:); 
    Best_FF=Ffun(ind);
    
    %Update the convergence curve
%     Conv_curve(C_Iter)=Best_FF;
%      
%     C_Iter=C_Iter+1;  % incremental iteration
   
end



