function model = maximization(X, R)
[d,n] = size(X);                                    
k = size(R,2);                                      
nk = sum(R,1);                                      
w = nk/n;                                                       
mu = bsxfun(@times, X*R, 1./nk);                   

Sigma = zeros(d,d,k);                               
r = sqrt(R);
for i = 1:k                                         
    Xo = bsxfun(@minus,X,mu(:,i));
    Xo = bsxfun(@times,Xo,r(:,i)');
    Sigma(:,:,i) = Xo*Xo'/nk(i)+eye(d)*(1e-6);
end

model.mu = mu;
model.Sigma = Sigma;
model.w = w;