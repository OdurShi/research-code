%Dynamic Fitness-Distance Balance (dFDB)
function index = dFDB( population, fitness, maxIter, iter )
[~, bestIndex] = min(fitness); 
best = population(bestIndex, :);
[populationSize, dimension] = size(population);

distances = zeros(1, populationSize); 
normFitness = zeros(1, populationSize); 
normDistances = zeros(1, populationSize); 
divDistances = zeros(1, populationSize); 

if min(fitness) == max(fitness)
    
    index = randi(populationSize);
    
else
    w = getParams(maxIter, iter);
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
        divDistances(i) = (1-w)*normFitness(i) + w*normDistances(i);
    end

    [~, index] = max(divDistances);

end

end

function w = getParams(maxIteration, iter)
    if iter>maxIteration
        w = 0.5;
    else
         f = 1;
         fx = round(maxIteration / f);
         y = mod(iter, fx);
         w = (y/fx * -0.6) + 0.6;
    end

end