function m = cauchy(var)
original_x=rand(1,var);
cauchy_x=tan((original_x-1/2)*pi);
m = cauchy_x;
end
