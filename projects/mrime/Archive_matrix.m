function C=Archive_matrix(X,A,t,l,num_elite)
% X:The current population
% A: Archive
% t:The current iteration number
% l: The archive length
% num_elite: The number of elites that are employed to construct the archive 

if t<l
    A=[A;X(1:num_elite,:)];
else
    A=[A;X(1:num_elite,:)];
    A(1:num_elite,:)=[];
end

mu_archive=mean(A);
sum1=0;
for i=1:size(A,1)
    sum1=sum1+((A(i,:)-mu_archive)')*(A(i,:)-mu_archive);
end

C=sum1/size(A,1);

end
