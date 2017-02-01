function [ trainY ] = kernPercGD( train_data )

data=load(train_data);
trainX=data(:,1:end-1);
r=data(:,end);

n=size(trainX,1);
rbf=zeros(n,n);
s=1;


for i=1:n
    for j=1:n
        rbf(j,i)=exp(norm(transpose(trainX(j,:)-trainX(i,:)))/(-2*s^2));
    end 
end

alpha=zeros(n,1);
discr=zeros(n,1);
k=1;
trainY=zeros(n,1);
delta_err=100;
pre_err= 0;

while abs(delta_err)>1
    for i=1:n
        discr(i,1)=sum(alpha.*r.*rbf(:,i))*r(i,1);
        if discr(i,1)<=0
            alpha(i,1)=alpha(i,1)+1;
        end
    end
    
    k=k+1;
    for i=1:n
        trainY(i,1)=sum(alpha.*r.*rbf(:,i));
    end
    err(k)= sum(sign(trainY)~=r);
    delta_err= err(k)-pre_err;
    pre_err=err;
end

end

