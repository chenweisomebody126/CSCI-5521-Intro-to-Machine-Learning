function[p1,p2,pc1,pc2] =Bayes_learning(training_data, validation_data)
% calculate the P_i_j as p1 and p2 for Bernoulli densities from training set
p1=zeros(size(training_data,2)-1,1)
p2=zeros(size(training_data,2)-1,1)
for j= 1: size(training_data,2)-1
    p1(j)=p1(j)+sum( (training_data(:,j)==0)& (training_data(:,end)==1) )/sum(sum(training_data(:,end)==1));
end
for j= 1: size(training_data,2)-1
    p2(j)=p2(j)+sum( (training_data(:,j)==0)& (training_data(:,end)==2) )/sum(sum(training_data(:,end)==2));
end

% calculate NB for each sigma. first initialize error rate matrix and prior
valid_err=zeros(11,1)
Prior=zeros(11,1)
for sigma=-5:5
% calculate the prior probability
    Prior(sigma+6)=Prior(sigma+6)+1/(1+exp(-sigma))


%calculate the posterior probability of valid set as a N by 2 matrix, N samples, 2
%classes
    P_c_x=ones(size(validation_data, 1),2)

    for n= 1:size(validation_data,1)
        for j=1:(size(validation_data,2)-1)
            P_c_x(n,1)=P_c_x(n,1)* p1(j).^(1-validation_data(n,j))* (1-p1(j)).^(validation_data(n,j))
        end
    end
    P_c_x(:,1)=P_c_x(:,1)*Prior(sigma+6)

    for n= 1:size(validation_data,1)
        for j=1:(size(validation_data,2)-1)
            P_c_x(n,2)=P_c_x(n,2)* p2(j).^(1-validation_data(n,j))* (1-p2(j)).^(validation_data(n,j))
        end
    end
    P_c_x(:,2)=P_c_x(:,2).*(1-Prior(sigma+6))

%compare c1 and c2 column to see the exact class as 3nd column
    for n=1:size(P_c_x,1)
        if P_c_x(n,1)>P_c_x(n,2)
            P_c_x(n,3)=1
        else
            P_c_x(n,3)=2
        end
    end
    
    
    valid_err(sigma+6)= valid_err(sigma+6)+sum(sum(P_c_x(:,3)~=validation_data(:,end)))/size(validation_data,1)
    
end
%find the index of minimium error rate, I, and then corresonding
%prior in class 1 and 2
[M,I]=min(valid_err)
pc1=Prior(I)
pc2=1-pc1
valid_err_print=sprintf('%0.5e',valid_err)
end

