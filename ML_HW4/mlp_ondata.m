function [z,w,v] = mlp_ondata(traindata,d,m,k)
%Let d= size of imput
%Let m= number of hidden units except bias unit
%Let k= size of output

X=traindata(:,1:end-1);

v=random('unif',-0.01,0.01,[k,m+1]);
w=random('unif', -0.01,0.01,[m,d+1]);
T=size(X,1);
z=zeros(T,m);
o=zeros(1,k);
y=zeros(1,k);

r=zeros(T,k);
for t=drange(1:T)
    r(t,traindata(t,end)+1)=1;
end
%add bias term as the first column of X
X=[ones(T,1) X];
delta_v=zeros(k,m+1);
delta_w=zeros(m,d+1);
%Initialize err to be any large number to strat while loop 
ita=0.001;
err=5;
prev_err=-1;
while abs(err-prev_err)> 5
%conver_err=zeros(1,100);
%for k=1:100
    prev_err=err;
    err=0;
    %loop through each observation t in random order
    for t=randperm(T)
        z_row=transpose(1./(1+exp(-w*transpose(X(t,:)))));
        z_row=[1 z_row];
        %Use softmax as output active function
        o=z_row*transpose(v);
        y=exp(o)/sum(exp(o));
        delta_v=ita*transpose((r(t,:)-y))*z_row;
        delta_w=ita*transpose(((r(t,:)-y)*v(:,2:end)).*z_row(2:end).*(1-z_row(2:end)))*X(t,:);

        v=v+delta_v;
        w=w+delta_w;
        %err=err+r(t,:)*transpose(y);
        err=err-r(t,:)*transpose(log(y));
        %conver_err(k)=conver_err(k)-r(t,:)*transpose(log(y));
        z(t,:)=z_row(2:end);
    end
    
end

end
