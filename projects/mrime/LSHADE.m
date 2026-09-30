function [best_fitness,best_solution, cg_curve] = LSHADE(population_size, max_iter,lower_limit, upper_limit, dim,fitness_function  )
    % Parameters
    memory_size = 5; % Size of the memory
    p_best_size = 5; % Number of best solutions considered for adaptation
    H = 6; % Scaling factor for population reduction
     cg_curve=zeros(1,max_iter);
    % Initialize population
    population = rand(population_size, dim)  .* (upper_limit - lower_limit) + lower_limit;
    fitness_values = fitness_function(population);

    % Initialize best solution and fitness
    [best_fitness, best_index] = min(fitness_values);
    best_solution = population(best_index, :);

    % Initialize memory
    memory = repmat(struct('F', 0.5, 'CR', 0.5), memory_size, 1);
iter=1;
    % Main loop
    while iter < max_iter+1
        % Generate trial solutions
        trial_population = zeros(population_size, dim);
        for i = 1:population_size
            % Select parent indices
            parent_indices = randperm(population_size, 3);

            % Select scaling factor and crossover rate
            memory_indices = randperm(memory_size, p_best_size);
            p_best_F = mean([memory(memory_indices).F]);
            p_best_CR = mean([memory(memory_indices).CR]);

            % Generate mutant vector
            F = max(0, normrnd(p_best_F, 0.1));
            CR = max(0, normrnd(p_best_CR, 0.1));
            mutant_vector = population(parent_indices(1), :) + F * (population(parent_indices(2), :) - population(parent_indices(3), :));

            % Crossover
            j_rand = randi(dim);
            trial_population(i, :) = population(i, :);
            for j = 1:dim
                if rand() <= CR || j == j_rand
                    trial_population(i, j) = mutant_vector(j);
                end
            end
        end

        % Evaluate trial solutions
        trial_fitness_values =fitness_function(trial_population);

        % Update population
        for i = 1:population_size
            if trial_fitness_values(i) < fitness_values(i)
                population(i, :) = trial_population(i, :);
                fitness_values(i) = trial_fitness_values(i);
                if trial_fitness_values(i) < best_fitness
                    best_fitness = trial_fitness_values(i);
                    best_solution = trial_population(i, :);
                end
            end
            
              
                                
 cg_curve(iter)=best_fitness;
   iter=iter+1;
                    if iter>max_iter
              break;
          end
            
        end

        % Update memory
        [~, sorted_indices] = sort(fitness_values);
        for i = 1:min(memory_size, population_size)
            memory(i) = struct('F', memory(i).F + 0.1 * (rand() - 0.5), 'CR', memory(i).CR + 0.1 * (rand() - 0.5));
        end
%          cg_curve(iter)=best_fitness;
    end

    % Population reduction
    population_size = max(2, round(population_size - H * iter / max_iter));
    population = population(sorted_indices(1:population_size), :);
    fitness_values = fitness_values(sorted_indices(1:population_size));

    % Return the best solution and fitness
    best_solution = population(1, :);
    best_fitness = fitness_values(1);
   
end
