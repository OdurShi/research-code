function [label, model, llh] = mixGaussEm(X, init)
tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);
R = initialization_FENZU(X,init);
for iter = 2:maxiter
    [~,label(1,:)] = max(R,[],2);     
    R = R(:,unique(label));                
    model = maximization(X,R);         
    [R, llh(iter)] = expectation(X,model);            
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end;
end
llh = llh(2:iter);