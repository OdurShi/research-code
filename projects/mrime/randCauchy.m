function y = randCauchy(mu, sigma,y_lub)
% mu = mean   sigma = Standard deviation
% http://en.wikipedia.org/wiki/Cauchy_distribution
 y_lb = y_lub(1);
 y_ub = y_lub(2);
 x_lb = (atan((y_lb - mu)/sigma) + 0.5) / pi;
 x_ub = (atan((y_ub - mu)/sigma) + 0.5) / pi;
 x = (x_ub-x_lb).* rand + x_lb;
 y = mu + sigma * tan(pi * x - 0.5);
end