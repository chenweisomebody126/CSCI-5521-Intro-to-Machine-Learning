function [ test_err_print] = Bayes_Testing(test_data, p1,p2,pc1,pc2 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%calculate the posterior probability of valid set as a N by 2 matrix, N samples, 2
%classes
P_c_x=ones(size(test_data, 1),2)

for n= 1:size(test_data,1)
    for j=1:(size(test_data,2)-1)
        P_c_x(n,1)=P_c_x(n,1)* p1(j).^(1-test_data(n,j))* (1-p1(j)).^(test_data(n,j))
    end
end
P_c_x(:,1)=P_c_x(:,1)*pc1

for n= 1:size(test_data,1)
    for j=1:(size(test_data,2)-1)
        P_c_x(n,2)=P_c_x(n,2)* p2(j).^(1-test_data(n,j))* (1-p2(j)).^(test_data(n,j))
    end
end
P_c_x(:,2)=P_c_x(:,2)*pc2
%compare c1 and c2 column to see the exact class as 3nd column
for n=1:size(P_c_x,1)
    if P_c_x(n,1)>P_c_x(n,2)
        P_c_x(n,3)=1
    else
        P_c_x(n,3)=2
    end
end
%calculate the error rate and print
test_err=sum(sum(P_c_x(:,3)~=test_data(:,end)))/size(test_data,1)
test_err_print=sprintf('%0.5e',test_err)
end

