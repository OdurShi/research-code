function y = loggausspdf(X, mu, Sigma)             
d = size(X,1);                                      
X = bsxfun(@minus,X,mu);                            
[U,p]= chol(Sigma);                                 
if p ~= 0                                           
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;                                          
q = dot(Q,Q,1);  
c = d*log(2*pi)+2*sum(log(diag(U)));   
y = -(c+q)/2;