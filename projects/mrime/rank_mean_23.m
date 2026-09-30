function [ RANK] = rank_mean_23(mmean)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
mmean=real(mmean);
[lll ~]=size(mmean);
for ii=1:lll
    ii;
AA=mmean(ii,:);%调整对比
BB = sort(AA);
[BB,Index] = sort(BB);
[C,I,J] = unique(BB);
E = Index - J';
for k = 1 : E(end)
temp = find(E == k);
E(temp(1)) = E(temp(1)) - 1;
end
F = E + J';
for i = 1:length(BB)
temp = find(BB==AA(i));
G(i) = temp(1);
end
RANK(ii,:)=G;
end


end

