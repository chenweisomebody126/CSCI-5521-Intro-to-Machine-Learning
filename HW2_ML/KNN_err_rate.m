function [ err_rate ] = KNN_err_rate( train_sam, test_sam, train_lab, test_lab, Kvals)
%To calculate the distance of each test observation to all training observation, 
%use pdist2 to calculate the distance of train matrix and test matrix as
%D_kNN
D=pdist2(train_sam,test_sam);
%To find the first k smallest distance, sort distance matrix and return original index
%in training dataset as I
[V,I]=sort(D);
% For each k, use mode func to capture the most frequently label in training dataset
% as the outcome label of test observation
MF=zeros(size(test_sam,1),1);
err_rate=zeros(1,length(Kvals));
for numk=1:length(Kvals)
    k=Kvals(numk);
    for i=1:size(test_sam,1)
        MF(i)=mode(train_lab(I(1:k,i)));     
    end
    err_rate(numk)=1-sum(MF==test_lab)/size(test_sam,1);
end

end

