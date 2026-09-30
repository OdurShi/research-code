%_____________________________________________________________________________________________%
%  source code:  MTV-SCA: Multi-trial Vector-based Sine Cosine Algorithm                      %
%                                                                                             %
%  Developed in: MATLAB R2018a                                                                %
% --------------------------------------------------------------------------------------------%
%  Main paper:   MTV-SCA: Multi-trial Vector-based Sine Cosine Algorithm                      %
%                Mohammad H Nadimi-Shahraki, Shokooh Taghian, Danial Javaheri,                %
%                Ali Safaa Sadiq, Nima Khodadadi, Seyedali Mirjalili                          %
%                                                                                             %
%  Emails:       nadimi@ieee.org, shokooh.taghian94@gmail.com, javaheri@korea.ac.kr           %
%                ali.sadiq@ntu.ac.uk, nima.khodadadi@miami.edu, ali.mirjalili@torrens.edu.au  %
%_____________________________________________________________________________________________%

function [gbestval,gbest,BestChart] = MTVSCA(N,MaxFES,lb,ub,D,fobj)
% rand('seed', sum(100 * clock));

%% Variables
FES = 0;
lambda = 0.25;
nImp = [1,1,1];
ImpRate = [0,0,0];

nIter = 20;
MaxIter = ceil(MaxFES/N);
winTVPIdx = 1;

BestChart=[]; 

% Define the dimension of the problem
lu = [lb .* ones(1, D); ub .* ones(1, D)];

%% Initialization
X = repmat(lu(1, :), N, 1) + rand(N, D) .* (repmat(lu(2, :) - lu(1, :), N, 1));
for i=1:N
    X_fitness(i) = fobj(X(i,:));
end
%-----------------------------------------
[gbestval,tmp1] = min(X_fitness);
gbest = X(tmp1, :);

Destination_position = gbest;
Destination_fitness = gbestval;

%%  %% ======================================================================%%%%
iter = 0;
nFES = [0,0,0];
repeat = floor(N / D);
copy = mod(N,D);

% Chaos coefficients
CC = chaos_coefficient(MaxIter);

%%
while  FES < MaxFES
    iter = iter +1;
    
    %%  %% ===========================Transformation Matrix:: M =====================================%%%%
    M_tril = tril(ones(D,D));   %lower triangular matrix
    if (copy == 0)&&(N ~= D)
        M_tril2= repmat(M_tril, repeat, 1);     
    else
        M_tril2 = repmat(M_tril, repeat, 1);
        added_row = M_tril(1:copy,:);
        M_tril2 = [M_tril2;added_row];
    end
    %for M
    Temp = M_tril2;
    Temp2 = zeros(N,D);
    
    %STEP1
    [nRows,nCols] = size(Temp);
    [~,idx] = sort(rand(nRows,nCols),2);
    idx = (idx-1)*nRows + ndgrid(1:nRows,1:nCols);
    Temp2(:) = Temp(idx);
    
    M_tmp = Temp2;
    M_step2 = M_tmp(randperm(size(M_tmp, 1)), :); %STEP 2:permute row of matrix
    
    M = M_step2;
    
    %for M_bar
    M_bar = ~M;
    
    %%  %% ===========================Distribution=====================================%%%%
    if mod(iter,nIter) == 0
        ImpRate(1) = nImp(1)/nFES(1);
        ImpRate(2) = nImp(2)/nFES(2);
        ImpRate(3) = nImp(3)/nFES(3);
        
        [~,winTVPIdx] = max(ImpRate);
        nImp = [1,1,1]; ImpRate =  [0,0,0]; nFES = [0,0,0];  
    end
    
    permutation = randperm(N);
    if winTVPIdx == 1
        array_PoolTVP2 = permutation(1:round(lambda*N));
        array_PoolTVP1 = permutation(round(lambda*N)+1:2*round(lambda*N));
        array_SCTVP = permutation(2*round(lambda*N)+1:end);
    elseif winTVPIdx == 2
        array_PoolTVP2 = permutation(1:round(lambda*N));
        array_SCTVP = permutation(round(lambda*N)+1:2*round(lambda*N));
        array_PoolTVP1  = permutation(2*round(lambda*N)+1:end);
    elseif winTVPIdx == 3
        array_SCTVP = permutation(1:round(lambda*N));
        array_PoolTVP1 = permutation(round(lambda*N)+1: 2*round(lambda*N));
        array_PoolTVP2  = permutation(2*round(lambda*N)+1:end);
    end
    nFES = nFES + [length(array_SCTVP),length(array_PoolTVP1),length(array_PoolTVP2)];
    %             end
    
    %%  %% ===========================Sine cosine trial vector producer (SC-TVP)=====================================%%%%
    if ~isempty(array_SCTVP)
        pop1 = X(array_SCTVP,:);
        fit1 = X_fitness(array_SCTVP);
        popsize1 = length(array_SCTVP);
        
        a = 2;
        r1=a-iter*((a)/MaxIter); % r1 decreases linearly from a to 0
        
        % Update the position of solutions with respect to destination
        for i=1:popsize1 % in i-th solution
            for j=1:D % in j-th dimension          
                % Update r2, r3, and r4
                r2=(2*pi)*rand();
                r3=2*rand;
                r4=rand();
                if r4<0.5
                    SCTVP_pop(i,j)= pop1(i,j)+(r1*sin(r2)*abs(r3*Destination_position(j)-pop1(i,j)));
                else
                    SCTVP_pop(i,j)= pop1(i,j)+(r1*cos(r2)*abs(r3*Destination_position(j)-pop1(i,j)));
                end
            end
        end
    end
    % -------------------------------------------------------------------------
    % Boundary checking
    SCTVP_pop = boundConstraint2(SCTVP_pop, lu);
    for i=1:popsize1
        SCTVP_fit(i) = fobj(SCTVP_pop(i,:));
    end
    
    
    tmp = (fit1 <= SCTVP_fit);
    temp1 = repmat(tmp',1,D);
    popnew1 = temp1 .* pop1 + (1-temp1) .* SCTVP_pop;
    
    nImp(1) = nImp(1) + sum(tmp == 0);
    
    X_trial(array_SCTVP,:) = popnew1;
    
    %%  %% ===========================Pool trial vector producer (Pool-TVP)=====================================%%%%
    if ~isempty(array_PoolTVP1)
        
        array_PoolTVP1 = [array_PoolTVP1,array_PoolTVP2];
        
        Pool_pop = X(array_PoolTVP1,:);
        Pool_fit = X_fitness(array_PoolTVP1);
        Pool_popsize = length(array_PoolTVP1);
        
        spara = randperm(4)';
        Str_num = randi([1,size(spara,1)],Pool_popsize,1);
        
        S1_x = find(Str_num==1); S2_x = find(Str_num==2); S3_x = find(Str_num==3); S4_x = find(Str_num==4);
        % == == == == == == == == == ==
        
        % if (Str_num(i,1)==1) % S1_TVP
        r0 = 1 : length(S1_x);
        popAll = X;
        [r1, r2] = gnR1R2(length(S1_x), length(S1_x), r0);
        [r3, r4] = gnR1R2(length(S1_x), length(S1_x), r0);
        
        Pool_TVP(S1_x,:) = Destination_position + CC.Chebyshev(iter)*(Pool_pop(r1,:)-Pool_pop(r2,:)) + rand.*(Pool_pop(r3,:)-Pool_pop(r4,:));
        Pool_TVP(S1_x,:) = M(array_PoolTVP1(S1_x),:) .* Pool_pop(S1_x,:) + M_bar(array_PoolTVP1(S1_x),:) .* Pool_TVP(S1_x,:);
        
        % if (Str_num(i,1)==2) % S2_TVP
        r0 = 1 : length(S2_x);
        popAll = X;
        [r1, r2] = gnR1R2(length(S2_x), length(S2_x), r0);
        [r3, r4] = gnR1R2(length(S2_x), length(S2_x), r0);
        
        Pool_TVP(S2_x,:) = Pool_pop(r1,:) + rand.*((Pool_pop(r2,:)-Pool_pop(r3,:)));
        Pool_TVP(S2_x,:) = M(array_PoolTVP1(S2_x),:) .* Pool_pop(S2_x,:) + M_bar(array_PoolTVP1(S2_x),:) .* Pool_TVP(S2_x,:);
        
        % if (Str_num(i,1)==3) % S3_TVP
        r0 = 1 : length(S3_x);
        popAll = X;
        [r1, r2] = gnR1R2(length(S3_x), length(S3_x), r0);
        [r3, r4] = gnR1R2(length(S3_x), length(S3_x), r0);
        
        Pool_TVP(S3_x,:) = Pool_pop(S3_x,:) + CC.Sinsudal(iter) .* (Pool_pop(r1,:)-Pool_pop(S3_x,:)) + rand .* (Pool_pop(r3,:)-Pool_pop(r4,:));
        
        % if (Str_num(i,1)==4) % S4_TVP
        r0 = 1 : length(S4_x);
        popAll = X;
        [r1, r2] = gnR1R2(length(S4_x), length(S4_x), r0);
        
        Pool_TVP(S4_x,:) = Pool_pop(S4_x,:) + (cos(iter/MaxIter).*sin(rand*(iter/MaxIter))).*(Pool_pop(r1,:)-Pool_pop(r2,:));
        
    end
% -------------------------------------------------------------------------
    % Boundary checking
    Pool_TVP = boundConstraint2(Pool_TVP, lu);
    
    for i=1:Pool_popsize
        Pool_TVP_fit(i) = fobj(Pool_TVP(i,:));
    end
    
    tmp = (Pool_fit <= Pool_TVP_fit);
    temp2 = repmat(tmp',1,D);
    popnew2 = temp2 .* Pool_pop + (1-temp2) .* Pool_TVP;
    
    nImp(2) = nImp(2) + sum(tmp == 0);
    
    X_trial(array_PoolTVP1,:) = popnew2;
    
    %% ================================================================%%%%    
    for i=1:N
        mixVal_trial(i) = fobj(X_trial(i,:));
    end
    
    bin = (mixVal_trial < X_fitness)';
    
    X(bin==1,:) = X_trial(bin==1,:);
    X_fitness(bin==1) = mixVal_trial(bin==1);
    
    [Destination_fitness, Idx] = min(X_fitness);
    Destination_position = X(Idx,:);
    
    for i=1:round(2*N)
    FES = FES + 1;    
    BestChart(FES) = Destination_fitness;
    
             if FES>=MaxFES
              break;
             end
    end
    clear SCTVP_pop Pool_TVP
    clear SCTVP_fit Pool_TVP_fit
    
end
%% ================================================================%%%%
gbestval = Destination_fitness;
gbest = Destination_position;
end
