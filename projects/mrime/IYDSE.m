function [Best_Fringe_fitness,BestFringe,cgcurve]=IYDSE(NP,Max_iter,LB,UB,Dim,F_obj)

beta1 = 3/2;
alfa = 0.05+0.04*rand;

L=1; % L is the distance between the barrier and the screen is often in the order of 1 m
d=5*10^-3; %distance between two slits FS and SS is often a fraction of millimeter
I=0.01; % distance between light source and barrier
Lambda=5*10^-6; % is the wavelength is a fraction of micrometer
Delta=0.38; % is a constant
%--------------------------------------------------------------------------
% Phase 1: initialize a source of Monochromatic light waves
%--------------------------------------------------------------------------
S=initialization_YDSE(NP,Dim,UB,LB); % Eq. (6) Initialize population  monochromatic light is projected S using Eq.(4) and Eq.(5)
%--------------------------------------------------------------------------
%Phase 2: Huygens' principle to generate the waves from the double-slits FS and SS
%--------------------------------------------------------------------------
FS=zeros(NP,Dim);
SS=zeros(NP,Dim);
S_Mean=mean(S);
for i=1:NP
    FS(i,:)= S(i,:)+I.*(2*rand-1).*(S_Mean-S(i,:)) ; %Eq. (9)
    SS(i,:)= S(i,:)-I.*(2*rand-1).*(S_Mean-S(i,:)) ; %Eq. (10)
end
%--------------------------------------------------------------------------
% Phase 3: travelling Two waves outgoing from the double slits : path difference
%--------------------------------------------------------------------------
BestFringe=zeros(1,Dim);   % the best-obtained so far solution
Best_Fringe_fitness = inf;      % the optimal Fitness value
Fringes_fitness=zeros(1,NP);
X=zeros(NP,Dim);
for i=1:NP
    if i==1   %m=i-1   then m=0   central fringe
        Delta_L=0;% AS THE TWO WAVES HAS THE SAME LEGTH  .. no path difference
        for j=1:Dim
            X(i,j)=(FS(i,j)+SS(i,j))/2+ Delta_L; %Eq.(12)
        end
    elseif mod(i,2)==0 %odd values of m=1, 3, 5, 7,...
        m=i-1;
        Delta_L=(2*m+1)*Lambda/2; % compute path difference for the odd m number
        for j=1:Dim
            X(i,j)= (FS(i,j)+SS(i,j))/2+Delta_L;%Eq.(12)
        end
    else            %even values of m=2,4,6,... compute path difference for the even m number
        m=i-1; %order number
        Delta_L=(m)*Lambda;
        for j=1:Dim
            X(i,j)= (FS(i,j)+SS(i,j))/2+Delta_L;%Eq.(12)
        end
    end
end
for i=1:NP
    F_UB=X(i,:)>UB;
    F_LB=X(i,:)<LB;
    X(i,:)=(X(i,:).*(~(F_UB+F_LB)))+UB.*F_UB+LB.*F_LB;
end
for i=1:NP
    Fringes_fitness(1,i) = F_obj(X(i,:));
end
%rank the solutions according to the fitness function from best to worst
[Fringes_fitness,sorted_indices]=sort(Fringes_fitness);
temp_X=X;
X=temp_X(sorted_indices,:);
Best_Fringe_fitness=Fringes_fitness(1,1);
BestFringe = X(1,:);
%--------------------------------------------------------------------------
%Phase 4: Interference pattern on the projection screen . bright and dark fringes
%--------------------------------------------------------------------------
Int_max0=10^-20;
iter=1;
X_New=zeros(NP,Dim);

while iter<Max_iter+1

    %rank the solutions according to the fitness function sort fitness by indices
    [Fringes_fitness,sorted_indices]=sort(Fringes_fitness);
    temp_X=X;
    X=temp_X(sorted_indices,:);
    q=iter/Max_iter;   %Eq.(26)
    Int_max=Int_max0.*q;   %Eq.(25)
    for  i=1:NP
        a=iter.^(2*rand()-1); %Eq.(20)
        H=2.*rand(1,Dim)-1;
        z=a./H;   %Eq.(19)
        r1=rand;
        %------------exploitation phase----------------
        if i==1 %zero m value
            beta=q.*cosh(pi./iter); %Eq.(15)
            A_bright=2./(1+sqrt(abs(1-beta.^2))); %Eq.(14)
            A=1:2:NP;
            randomIndex = randi(length(A));%select random bright fringe
            X_New(i,:)=BestFringe+(Int_max.*A_bright.*X(i,:)-r1.*z.*X(A(randomIndex),:)); % %Eq. (24)
        elseif  mod(i,2)==1 %even m values
            beta=q.*cosh(pi./iter);%Eq.(15)
            A_bright=2./(1+sqrt(abs(1-beta.^2)));  %Eq.(14)
            m=i-1;
            y_bright=Lambda*L*m/d; %Eq.(4)
            Int_bright=Int_max.*cos((pi*d)/(Lambda*L)*y_bright).^2;   %Eq.(23)  %update light intensity using Eq.(18)
            s=randperm(NP,2);
            g=2*rand-1;
            Y=(X(s(2),:)-X(s(1),:)); %select two random fringes Eq.(22)
            X_New(i,:)=X(i,:)-((1-g).*A_bright.*Int_bright.*X(i,:)+g.*Y);   %Eq.(21)
            
            %------------exploration-------------------
        else % odd m values
            A_dark=Delta*atanh(-(q)+1); %Eq.(16)
            m=i-1;
            y_dark=Lambda*L*(m+0.5)/d; %Eq.(5)
            Int_dark=Int_max.*cos((pi*d)/(Lambda*L)*y_dark).^2;%Eq.(18)
            X_New(i,:)=X(i,:)-(alfa*levyrand(beta1).*A_dark.*Int_dark.*X(i,:)-z.*BestFringe); %Eq. (17)
        end
        %check bounds of new solutions
        F_UB=X_New(i,:)>UB;
        F_LB=X_New(i,:)<LB;
        X_New(i,:)=(X_New(i,:).*(~(F_UB+F_LB)))+UB.*F_UB+LB.*F_LB;
    end
    %Generate the next population
    X_New_Fitness=zeros(1,size(X,1));
    for i=1:NP
        X_New_Fitness(1,i) = F_obj(X_New(i,:));
        if X_New_Fitness(1,i)<Fringes_fitness(1,i)
            X(i,:)=X_New(i,:);
            Fringes_fitness(1,i)=X_New_Fitness(1,i);
            if Fringes_fitness(1,i) < Best_Fringe_fitness
                Best_Fringe_fitness=Fringes_fitness(1,i);
                BestFringe = X(i,:);
            end
        end
        
      
                       iter=iter+1;  
       cgcurve(iter)=Best_Fringe_fitness;
         if iter>Max_iter
              break;
          end
        
        
    end
    
    FitnessScores2=zeros(1,size(X,1));
    if rand<0.1
        varmin=min(X);
        varmax=max(X);
        
        for i=1:NP
            if rand<rand
                lamda=1+rand*rand;
            else
                lamda=1-rand*rand;
            end
            XX(i,:)=(0.5+0.5*lamda)*(varmin+varmax)-lamda*X(i,:);
            F_UB=XX(i,:)>UB;
            F_LB=XX(i,:)<LB;
            XX(i,:)=(XX(i,:).*(~(F_UB+F_LB)))+UB.*F_UB+LB.*F_LB;
            FitnessScores2(i)= F_obj(XX(i,:));
            if FitnessScores2(1,i)<Fringes_fitness(1,i)
                X(i,:)=XX(i,:);
                Fringes_fitness(1,i)=FitnessScores2(1,i);
                if Fringes_fitness(1,i) < Best_Fringe_fitness
                    Best_Fringe_fitness=Fringes_fitness(1,i);
                    BestFringe = X(i,:);
                end
            end
            
                 
                       iter=iter+1;  
       cgcurve(iter)=Best_Fringe_fitness;
         if iter>Max_iter
              break;
          end  
            
            
        end
    end
    
    for  i=1:2:NP-1
        beta=q.*cosh(pi./iter);%Eq.(15)
        A_bright=2./(1+sqrt(abs(1-beta.^2)));  %Eq.(14)
        A_dark=Delta*atanh(-(q)+1); %Eq.(16)
        C1=2*rand*(1-iter/Max_iter);
        X_New(i,:)=rand*A_bright*X(i,:)+rand*A_dark*X(i+1,:)+C1*alfa*levyrand(beta1)*(X(i,:)-X(i+1,:));   %Eq.(21)
        X_New(i+1,:)=rand*A_bright*X(i+1,:)+rand*A_dark*X(i,:)+C1*alfa*levyrand(beta1)*(X(i+1,:)-X(i,:));   %Eq.(21)
        F_UB=X_New(i,:)>UB;
        F_LB=X_New(i,:)<LB;
        X_New(i,:)=(X_New(i,:).*(~(F_UB+F_LB)))+UB.*F_UB+LB.*F_LB;
        F_UB=X_New(i+1,:)>UB;
        F_LB=X_New(i+1,:)<LB;
        X_New(i+1,:)=(X_New(i+1,:).*(~(F_UB+F_LB)))+UB.*F_UB+LB.*F_LB;
    end
    X_New_Fitness=zeros(1,size(X,1));
    for i=1:NP
        X_New_Fitness(1,i) = F_obj(X_New(i,:));
        if X_New_Fitness(1,i)<Fringes_fitness(1,i)
            X(i,:)=X_New(i,:);
            Fringes_fitness(1,i)=X_New_Fitness(1,i);
            if Fringes_fitness(1,i) < Best_Fringe_fitness
                Best_Fringe_fitness=Fringes_fitness(1,i);
                BestFringe = X(i,:);
            end
        end
             
                       iter=iter+1;  
       cgcurve(iter)=Best_Fringe_fitness;
         if iter>Max_iter
              break;
          end
        
    end
 
end
end