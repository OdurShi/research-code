function sort_index = fitnessDistanceBalanceWithWeightGaussian(population, fitness)

    [~, bestIndex] = min(fitness); 
    best = population(bestIndex, :);
    [populationSize, dimension] = size(population);

    distances = zeros(1, populationSize); 
    normFitness = zeros(1, populationSize); 
    normDistances = zeros(1, populationSize); 
    divDistances = zeros(1, populationSize); 
    fd = zeros(1, populationSize);
%% Set time weight w
    %w = 0.2 + 0.6*(nFES/FES); w = vpa(w,4); %case1
    
    %w = rand; % Randomly generate a time weight w uniformly distributed in the interval (0,1)
    %case2
%     w = normrnd(0.5,1/6);
     w = normrnd(0.75,1/12); % Randomly generate a time weight w with a standard normal distribution in the interval (0, 1), the interval (0, 1) can be changed
    %case3
%      fx = round(Gen / frequency);
%  y = mod(iter, fx);
%  w = (y/fx * -0.6) + 0.6;
% w = (y/fx * -0.2) + 0.2;
    % w_rand = [0.2,0.4,0.6,0.8]; w_rand = w_rand(randperm(length(w_rand))); w = w_rand(1);  % Randomly select the time weight w in an array
    %case4
%%    
    if min(fitness) == max(fitness)
        sort_index = randperm(populationSize, populationSize);
        
    else

        for i = 1 : populationSize
            value = 0;
            for j = 1 : dimension
                value = value + abs(best(j) - population(i, j));
            end
            distances(i) = value;
        end

        minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
        minDistance = min(distances); maxMinDistance = max(distances) - minDistance;

        for i = 1 : populationSize
            normFitness(i) = 1 - ((fitness(i) - minFitness) / maxMinFitness);
            normDistances(i) = (distances(i) - minDistance) / maxMinDistance;
            divDistances(i) = w * normFitness(i) + (1-w) * normDistances(i);
        end
%         for i = 1 : populationSize
%             normFitness(i) = mapminmax((fitness(i) - minFitness), 0, 1);
%             normDistances(i) = mapminmax((distances(i) - minDistance) , 0, 1);
%             divDistances(i) = normFitness(i) + normDistances(i);
%         end
%             dd = [];
         fd = mapminmax(divDistances, 0, 1);
         [~,sort_index] = sort(fd, 'descend');

%         [~, index] = max(dd);
    end
end

