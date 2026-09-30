%% **************************** EOSMA *****************************
% Algorithm disadvantages:
% 1. The algorithm parameters cannot be adaptively selected, 
%    and the selection of parameter z=0.6 depends on the problem solved
% 2. The number of iterations needs to be given beforehand, 
%    because the parameter in the algorithm is a function of the number of iterations
% 3. As the number of iterations increases, it is easy to fall into local optimum
% Algorithm advantages:
% 1. Fast convergence speed, can quickly obtain high precision feasible solution, 
%    suitable for real-time system
% 2. The principle is simple and easy to implement, no archiving mechanism, 
%    low space and time complexity
% 3. Strike a better balance between exploration and exploitation
%% ****************************************************************
% Author: Shihong Yin, Qifang Luo, and Yongquan Zhou. 
% "EOSMA: An Equilibrium Optimizer Slime Mould Algorithm for Engineering Design Problems."
% Arabian Journal for Science and Engineering (2022): 1-32.
%% ****************************************************************
function [Destination_fitness,bestPositions,Convergence_curve] = EOSMA(PopSize,Max_iter,lb,ub,dim,fobj)
% rand('seed',sum(100 * clock)); % Random number seed
V = 1;  a1 = 2;  a2 = 1;  GP = 0.5; % EO parameters
z = 0.3; % SMA parameter
q = 0.2; % Mutation parameter
% Initialize the position of slime mould
lb = ones(1,dim).*lb; % Lower boundary
ub = ones(1,dim).*ub; % Upper boundary
X = initialization(PopSize,dim,ub,lb); % It can be downloaded from https://github.com/Shihong-Yin
Fitness = inf.*ones(PopSize,1);
weight = ones(PopSize,dim); % Fitness weight of each slime mould
% Initializes the equilibrium pool
bestPositions = zeros(1,dim);  Destination_fitness = inf;
Ceq2 = zeros(1,dim);  Ceq2_fit = inf;
Ceq3 = zeros(1,dim);  Ceq3_fit = inf;
Ceq4 = zeros(1,dim);  Ceq4_fit = inf;
Xold = X;  Xold_fit = Fitness;
Convergence_curve = zeros(1,Max_iter);
% Main loop
it=1;
while it < 1+Max_iter
    % Check the boundary and calculate the fitness
    FU = X>ub;  FL = X<lb;  X = (X.*(~(FU+FL)))+(Xold+ub)./2.*FU+(Xold+lb)./2.*FL;
    Fitness = fobj(X); % https://github.com/Shihong-Yin
    % Update the fitness values
    for i = 1:PopSize
        if Xold_fit(i) < Fitness(i)
            Fitness(i) = Xold_fit(i);  X(i,:) = Xold(i,:);
        end
    end
    Xold = X;  Xold_fit = Fitness;
    % Sort the fitness thus update the bF and wF
    [SmellOrder,SmellIndex] = sort(Fitness);
    bestFitness = SmellOrder(1);
    worstFitness = SmellOrder(PopSize);
    S = bestFitness-worstFitness+eps; % Plus eps to avoid denominator zero
    % Calculate the fitness weight of each slime mould
    for i = 1:PopSize
        if i <= PopSize/2
            weight(SmellIndex(i),:) = 1+rand(1,dim)*log10((bestFitness-SmellOrder(i))/S+1);
        else
            weight(SmellIndex(i),:) = 1-rand(1,dim)*log10((bestFitness-SmellOrder(i))/S+1);
        end
    end
    % Update the equilibrium pool
    for i = 1:PopSize 
        if Fitness(i)<Destination_fitness 
            Destination_fitness = Fitness(i);  bestPositions = X(i,:);
        elseif Fitness(i)>Destination_fitness && Fitness(i)<Ceq2_fit
            Ceq2_fit = Fitness(i);  Ceq2 = X(i,:);
        elseif Fitness(i)>Destination_fitness && Fitness(i)>Ceq2_fit && Fitness(i)<Ceq3_fit
            Ceq3_fit = Fitness(i);  Ceq3 = X(i,:);
        elseif Fitness(i)>Destination_fitness && Fitness(i)>Ceq2_fit && Fitness(i)>Ceq3_fit && Fitness(i)<Ceq4_fit
            Ceq4_fit = Fitness(i);  Ceq4 = X(i,:);
        end
        
        Convergence_curve(it)=Destination_fitness;
        it=it+1;
                                if it>Max_iter
              break;
                                end
        
    end
    Ceq_ave = (bestPositions + Ceq2 + Ceq3 + Ceq4)/4; % Average candidate solution
    C_pool = [bestPositions; Ceq2; Ceq3; Ceq4; Ceq_ave]; % Equilibrium pool
    % Update EO parameters
    t = (1-it/Max_iter)^(a2*it/Max_iter);
    lambda = rand(PopSize,dim);
    r1 = rand(PopSize,1);
    r2 = rand(PopSize,1);
    rn = randi(size(C_pool,1),PopSize,1);
    % Update SMA parameters
    a = atanh(1-(it/Max_iter));
    vb = unifrnd(-a,a,PopSize,dim);
    b = 1-it/Max_iter;
    vc = unifrnd(-b,b,PopSize,dim);
    p = tanh(abs(Fitness-Destination_fitness));
    r = rand(PopSize,dim);
    A = randi([1,PopSize],PopSize,dim); % Two positions randomly selected from population
    B = randi([1,PopSize],PopSize,dim);
    % Update the position of search agents
    for i = 1:PopSize
        if rand < z
            Ceq = C_pool(rn(i,1),:); % Select a solution randomly from the equilibrium pool
            F = a1*sign(r(i,:)-0.5).*(exp(-lambda(i,:).*t)-1);
            GCP = 0.5*r1(i,1)*ones(1,dim)*(r2(i,1)>=GP);
            G = (GCP.*(Ceq-lambda(i,:).*X(i,:))).*F;
            X(i,:) = Ceq+(X(i,:)-Ceq).*F+(G./lambda(i,:)*V).*(1-F);
        else
            for j = 1:dim
                if r(i,j) < p(i)
                    X(i,j) = X(i,j)+vb(i,j)*(weight(i,j)*X(A(i,j),j)-X(B(i,j),j)); % Overcome premature convergence
                else
                    X(i,j) = bestPositions(j)+vc(i,j)*(weight(i,j)*X(A(i,j),j)-X(B(i,j),j));
                end
            end
        end
    end
    % Check the boundary and calculate the fitness
    FU = X>ub;  FL = X<lb;  X = (X.*(~(FU+FL)))+(Xold+ub)./2.*FU+(Xold+lb)./2.*FL;
    %Fitness = cec21_bias_shift_rot_func(X',fobj)';
    Fitness=fobj(X);
    % Update the fitness values
    for i = 1:PopSize
        if Xold_fit(i) < Fitness(i)
            Fitness(i) = Xold_fit(i);  X(i,:) = Xold(i,:);
        end
    end
    Xold = X;  Xold_fit = Fitness;
    
       for i = 1:PopSize 
        if Fitness(i)<Destination_fitness 
            Destination_fitness = Fitness(i);  bestPositions = X(i,:);
        elseif Fitness(i)>Destination_fitness && Fitness(i)<Ceq2_fit
            Ceq2_fit = Fitness(i);  Ceq2 = X(i,:);
        elseif Fitness(i)>Destination_fitness && Fitness(i)>Ceq2_fit && Fitness(i)<Ceq3_fit
            Ceq3_fit = Fitness(i);  Ceq3 = X(i,:);
        elseif Fitness(i)>Destination_fitness && Fitness(i)>Ceq2_fit && Fitness(i)>Ceq3_fit && Fitness(i)<Ceq4_fit
            Ceq4_fit = Fitness(i);  Ceq4 = X(i,:);
        end
        
        Convergence_curve(it)=Destination_fitness;
        it=it+1;
                                if it>Max_iter
              break;
                                end
                Convergence_curve(it)=Destination_fitness;
        it=it+1;
                                if it>Max_iter
              break;
                                end
    end 
    
    
    
    % Random difference mutation operator
    if rand < q
        L = rand(PopSize,dim)<q;
        CF = (1-it/Max_iter)^(2*it/Max_iter); % Contraction factor
        X = X+CF*((lb+rand(PopSize,dim).*(ub-lb)).*L);
    else
        R = 0.2+rand*(1-0.2);
        X = X+R*(X(randperm(PopSize),:)-X(randperm(PopSize),:));
    end
%     Convergence_curve(it) = Destination_fitness;
end
end
% Developer: Shihong Yin