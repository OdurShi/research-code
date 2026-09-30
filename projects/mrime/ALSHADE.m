%The MATLAB source code of AL-SHADE.
%Please cite this article as: 
%Yintong Li, Tong Han, Huan Zhou, Shangqin Tang, Hui Zhao, 
%A novel adaptive L-SHADE algorithm and its application in UAV swarm resource configuration problem, Information Sciences, https://doi.org/10.1016/j.ins.2022.05.058
function [Global_score,Global_pos,Convergence_curve] = ALSHADE(SearchAgents_no,FEsmax,lb,ub,dim,fobj)
lb=lb.*ones(1,dim);
ub=ub.*ones(1,dim);
% Initialize parameter
F = 0.5;%MCR,k, MF,k (k = 1, ..., H) are all initialized to 0.5
CR = 0.5;%MCR,k, MF,k (k = 1, ..., H) are all initialized to 0.5
rarc = 2.6; %external archive size |A| = round(Ninit ¡Á rarc)
p = 0.11; %xpbest,G is randomly selected from the top N ¡Á p (p ¡Ê [0, 1]) members in generation G
H = 6;   %historical memory size H.
NPinit = SearchAgents_no;%initial population size
NPmin = 4;%the population number at the end of the run
counteval = 0;
countiter = 1;
P=0.5;
% Initialize population
X = lb + (ub - lb) .* rand(SearchAgents_no,dim );
X=X';
fitness = fobj(X');

% Sort
[fitness, fidx] = sort(fitness);
X = X(:, fidx);
% Initialize archive
Asize = round(rarc * SearchAgents_no);%archive size |A| = round(Ninit ¡Á rarc)
A=[];
nA=0;
MF = F * ones(H, 1);%H :historical memory size H.
MCR = CR * ones(H, 1);
MCR(H)=0.9;
MF(H)=0.9;
iM = 1;
% Initialize variables
A(:, nA + 1)	= X(:,1);
Afitness(nA + 1)=fitness(1);
nA= nA + 1;
V = X;
U = X;
S_CR = zeros(1, SearchAgents_no);	% Set of crossover rate
S_F = zeros(1, SearchAgents_no);		% Set of scaling factor
S_df = zeros(1, SearchAgents_no);	% Set of df
%Generate random numbers from the Cauchy distribution, r= a + b*tan(pi*(rand(n)-0.5)).
Chy = cauchyrnd(0, 0.1, SearchAgents_no+ 200);
iChy = 1;
while counteval < FEsmax
    SEL=ceil(nA/2);
    weights = log(SEL+1/2)-log(1:SEL)';
    weights = weights/sum(weights);
    Xsel = A(:,1:SEL);
    xmean   = Xsel*weights;
    % pbest index
    pbest = 1 + floor(max(2, round(p * SearchAgents_no)) * rand(1, SearchAgents_no));
    % Memory Indices
    r = floor(1 + H * rand(1, SearchAgents_no));
    % Crossover rates
    CR = MCR(r)' + 0.1 * randn(1, SearchAgents_no);
    CR((CR < 0) | (MCR(r)' == -1)) = 0;
    CR(CR > 1) = 1;
    % Scaling factors
    F = zeros(1, SearchAgents_no);
    for i = 1 : SearchAgents_no
        while F(i) <= 0
            F(i) = MF(r(i)) + Chy(iChy);
            iChy = mod(iChy, numel(Chy)) + 1;
        end
    end
    F(F > 1) = 1;
    PA = [X, A];
    % Mutation
    memory=zeros(1,SearchAgents_no);
    for i = 1 : SearchAgents_no
        % Generate r1
        r1 = floor(1 + SearchAgents_no * rand);
        while i == r1
            r1 = floor(1 + SearchAgents_no * rand);
        end
        % Generate r2
        r2 = floor(1 + (SearchAgents_no + nA) * rand);
        while i == r1 || r1 == r2
            r2 = floor(1 + (SearchAgents_no + nA) * rand);
        end
        if rand<P
            V(:, i) = X(:, i)+F(i) .* (X(:, pbest(i)) - X(:, i)) + F(i) .* (X(:, r1) - PA(:, r2));
            memory(i)=1;
        else
            V(:, i) = X(:, i) + F(i) .* (xmean - X(:, i)) + F(i) .* (X(:, r1) - PA(:, r2));
            memory(i)=3;
        end
        % Correction for outside of boundaries
        for j = 1 : dim
            if V(j, i) < lb(j)
                V(j, i) = 0.5 * (lb(j) + X(j, i));
            end
            if V(j, i) > ub(j)
                V(j, i) = 0.5 * (ub(j) + X(j, i));
            end
        end
        % Binomial Crossover
        jrand = floor(1 + dim * rand);
        for j = 1 : dim
            if rand < CR(i) || j == jrand
                U(j, i) = V(j, i);
            else
                U(j, i) = X(j, i);
            end
        end
    end
    % Evaluation
    fu=fobj(U');
%     counteval = counteval + SearchAgents_no;
    % Selection
    elitism=fu<=fitness;
    LL=zeros(1,SearchAgents_no);
    LL(elitism)=1;
    LLL=memory+LL;
    
    A1_ALL=sum(find(memory==1));
    A1_better=sum(find(LLL==2));
    
    A2_ALL=sum(find(memory==3));
    A2_better=sum(find(LLL==4));
    
    if A1_ALL~=0 && A2_ALL~=0
        P_A1=A1_better/A1_ALL; 
        P_A2=A2_better/A2_ALL; 
        P=P+0.05*(1-P)*(P_A1-P_A2)*counteval/FEsmax;
        P=min(0.9,P);
        P=max(0.1,P);
    end
    nS = 0;
    for i = 1 : SearchAgents_no
        if fu(i) < fitness(i)
            nS			= nS + 1;
            S_CR(nS)	= CR(i);
            S_F(nS)		= F(i);
            S_df(nS)	= abs(fu(i) - fitness(i));
            X(:, i)		= U(:, i);
            fitness(i)		= fu(i);
            if nA < Asize
                A(:, nA + 1)	= X(:, i);
                Afitness(nA + 1)=fu(i);
                nA= nA + 1;
            else
                ri= floor(1 + Asize * rand);
                A(:, ri)= X(:, i);
                Afitness(ri)=fu(i);
            end
        elseif fu(i) == fitness(i)
            X(:, i)		= U(:, i);
        end
    end
    % Update MCR and MF
    if nS > 0
        w = S_df(1 : nS) ./ sum(S_df(1 : nS));
        if all(S_CR(1 : nS) == 0)
            MCR(iM) = -1;
        elseif MCR(iM) ~= -1
            MCR(iM) = sum(w .* S_CR(1 : nS) .* S_CR(1 : nS)) / sum(w .* S_CR(1 : nS));
        end
        MF(iM) = sum(w .* S_F(1 : nS) .* S_F(1 : nS)) / sum(w .* S_F(1 : nS));
        iM = mod(iM, H-1) + 1;
    end
    % Sort
    [fitness, fidx] = sort(fitness);
    X = X(:, fidx);
    % Update NP and population
    SearchAgents_no = round(NPinit - (NPinit - NPmin) * counteval / FEsmax);
    fitness = fitness(1 : SearchAgents_no);
    X = X(:, 1 : SearchAgents_no);
    U = U(:, 1 : SearchAgents_no);
    [Afitness, Ax] = sort(Afitness);
    A = A(:, Ax);
    Asize = round(rarc * SearchAgents_no);
    if nA > Asize
        nA = Asize;
        A = A(:, 1 : Asize);
        Afitness=Afitness(1 : Asize);
    end
%     counteval < FEsmax
        for i=1:SearchAgents_no
    counteval = counteval + 1;    
    Convergence_curve(counteval) = fitness(1);
    
             if counteval>=FEsmax
              break;
             end
    end
    
%     Convergence_curve(1,countiter)=fitness(1);
%     countiter = countiter + 1;
end
Global_score = fitness(1);
Global_pos = X(:, 1);
