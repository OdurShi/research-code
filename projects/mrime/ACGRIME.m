function [Best_rime_rate,Best_rime, Convergence_curve] = ACGRIME(N, MaxFEs, lb, ub, dim, fobj)
 
    Best_rime = zeros(1, dim);
    Best_rime_rate = inf; % change this to -inf for maximization problems
    Rimepop = initialization(N, dim, ub, lb);
 
    Lb = lb .* ones(1, dim);
    Ub = ub .* ones(1, dim);
    FEs = 0; % Number of function evaluations
    Convergence_curve = [];
    Rime_rates = zeros(1, N);
    newRime_rates = zeros(1, N);
    Time =1;
    ChaosValue = rand(); % Initial value for Logistic map
    a = 4; % Parameter for Logistic map
 
    for i = 1:N
        Rime_rates(1, i) = fobj(Rimepop(i, :));
        FEs = FEs + 1;
        if Rime_rates(1, i) < Best_rime_rate
            Best_rime_rate = Rime_rates(1, i);
            Best_rime = Rimepop(i, :);
        end
    end
 
    while FEs < MaxFEs
        % Apply Logistic map for chaos
        ChaosValue = a * ChaosValue * (1 - ChaosValue);
        E = sqrt(ChaosValue * FEs/MaxFEs);
        newRimepop = Rimepop;
        normalized_rime_rates = normr(Rime_rates);
        w1 = (1-FEs/MaxFEs)^(1-tan(pi*(ChaosValue * rand-0.5))*FEs/MaxFEs); % Adaptive weighting
 
        for i = 1:N
            for j = 1:dim
                r1 = rand();
                r2 = rand();
                r3 = rand();
                c1 = 2*exp(-(4*FEs/MaxFEs)^2);
                c2 = rand();
 
                if r2 < E
                    if c2 < 0.5
                        newRimepop(i, j) = Best_rime(1, j) + w1 * c1 * ((Ub(j) - Lb(j)) * r1 + Lb(j));
                    else
                        newRimepop(i, j) = Best_rime(1, j) - w1 * c1 * ((Ub(j) - Lb(j)) * r1 + Lb(j));
                    end
                end

 
                if r3 < normalized_rime_rates(i)
                    newRimepop(i, j) = Best_rime(1, j);
                end
            end
 
            % Gaussian Mutation
            x = newRimepop(i, :);
            m_gaus = x * (1 + ChaosValue * randn(1));
            Flag4ub = m_gaus > ub;
            Flag4lb = m_gaus < lb;
            m_gaus = (m_gaus .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            m_gaus_fitness = fobj(m_gaus);
            fitness_s = fobj(x);
            m_gaus_fitness_comb = [m_gaus_fitness, fitness_s];
            [~, m] = min(m_gaus_fitness_comb);
            if m == 1
                newRimepop(i, :) = m_gaus;
            end
        end
        
        for i = 1:N
            Flag4ub = newRimepop(i, :) > ub;
            Flag4lb = newRimepop(i, :) < lb;
            newRimepop(i, :) = (newRimepop(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            newRime_rates(1, i) = fobj(newRimepop(i, :));
           
 
            if newRime_rates(1, i) < Rime_rates(1, i)
                Rime_rates(1, i) = newRime_rates(1, i);
                Rimepop(i, :) = newRimepop(i, :);
                if newRime_rates(1, i) < Best_rime_rate
                    Best_rime_rate = Rime_rates(1, i);
                    Best_rime = Rimepop(i, :);
                end
            end
            
                 FEs = FEs + 1;          
       Convergence_curve(FEs)=Best_rime_rate; 
         if FEs>MaxFEs
              break;
          end
            
                            FEs = FEs + 1;          
       Convergence_curve(FEs)=Best_rime_rate; 
         if FEs>MaxFEs
              break;
          end 
        end
 
%         Convergence_curve(Time)=Best_rime_rate;
%         Time=Time+1;
    end
end

