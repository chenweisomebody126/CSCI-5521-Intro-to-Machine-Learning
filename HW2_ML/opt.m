opt_train=load('optdigits_train.txt');
opt_test=load('optdigits_test.txt');
%Q2(a)
err_rate= KNN_err_rate(opt_train,opt_test,opt_train(:,end),opt_test(:,end),[1,3,5,7]);
disp(err_rate);

%Q2(b)
%first calculate the covariance matrix of attributes(remove label)
cov_train=cov(opt_train(:,1:end-1));
[V_train,lambda_train] = eig(cov_train);
%Note that eigenvector given by eig func is n ascending order, we need to
%choose the largest eigenvalue which is the last one to project new
%coordinates project_train and project_project_test
project_train=opt_train(:,1:end-1)*V_train(:,end-1:end);
project_test=opt_test(:,1:end-1)*V_train(:,end-1:end);
% Calculate the error rate
err_rate_opt_project= KNN_err_rate(project_train,project_test,opt_train(:,end),opt_test(:,end),[1,3,5,7]);
disp(err_rate_opt_project);
%plot three graph corresponding to k=1,3,5
boundary(project_test, opt_test(:,end),[1 3 5])

%Q2(c)
%To calculate the S_w within-class scatter, initialize the dimension looping 
%from 64
dim=64;
%initialize cov_loop as a uninversible matrix to start the loop
S_w=zeros(1);
V_train_descend=fliplr(V_train);
while rank(S_w)<size(S_w,1)  
   project_loop=opt_train(:,1:end-1)*V_train_descend(:,1:(dim-1)); 
   S_w=zeros(dim-1);
   for c=0:9
       S_w=S_w+cov(project_loop(opt_train(:,end)==c,:));    
   end
   dim=dim-1;
end
% Calculate the between_class scatter
global_mean=mean(project_loop);
S_b=zeros(size(global_mean,2));
for c=0:9
    class_mean=mean(project_loop(opt_train(:,end)==c,:));
    S_b=S_b+sum(opt_train(:,end)==c)*transpose(class_mean-global_mean)*(class_mean-global_mean);
end
%compute a projection using LDA that is largest(last) eigenvectors of 
%inv(S_w)*S_b
[vector_LDA,lambda_LDA]=eig((inv(S_w))*(S_b));
disp(vector_LDA);
%Use PCA to reduce dimentin to dim=61 and project to N by dim matrix,
%project_both
opt_both=[opt_train; opt_test];
cov_both=cov(opt_both(:, 1:end-1));
[V_both, L_both]=eig(cov_both);
V_both_descend=fliplr(V_both);
project_both=opt_both(:,1:end-1)*V_both_descend(:,1:dim);
%Use LDA's last two eigenvector(two largest) to project to N by 2 matrix 
project_LDA=project_both*vector_LDA(:,end-1:end);
%Run kNN on projected data for k={1,3,5,7}
project_err_rate=KNN_err_rate(project_LDA(1:size(opt_train),1:end),project_LDA(end-size(opt_test,1)+1:end,1:end),opt_train(:,end),opt_test(:,end),[1,3,5,7]);
disp(project_err_rate);
%plot three graph corresponding to k=1,3,5
boundary(project_LDA(end-size(opt_test,1)+1:end,:), opt_test(:,end),[1 3 5])

