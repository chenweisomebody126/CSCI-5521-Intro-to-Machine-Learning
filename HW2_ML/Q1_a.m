training=load('training_data.txt');
training_c1=training((training(:,end)==1),:);
training_c2=training((training(:,end)==2),:);
test=load('test_data.txt');
%all three questions (a)(b)(c) have same mean estimate
mu_1= mean(training_c1(:,1:end-1))
mu_2=mean(training_c2(:,1:end-1))

%The covariance is different three questions
%In question (a)
S1_a=cov(training_c1(:,1:end-1)) 
S2_a=cov(training_c2(:,1:end-1))

postp1_a = mvnpdf(test(:,1:end-1),mu_1,S1_a);
postp2_a = mvnpdf(test(:,1:end-1),mu_2,S2_a);
count_a=0
for i=1:100 
    if (postp1_a(i,1)>postp2_a(i,1)) & test(i,end)==1
        count_a=count_a+1
    elseif (postp1_a(i,1)<postp2_a(i,1)) & test(i,end)==2
        count_a=count_a+1
    end 
end
err_rate_a=1-count_a/100
%In question (b)
S1_b=cov(training(:,1:end-1))
S2_b=S1_b
postp1_b = mvnpdf(test(:,1:end-1),mu_1,S1_b);
postp2_b = mvnpdf(test(:,1:end-1),mu_2,S2_b);
count_b=0
for i=1:100 
    if (postp1_b(i,1)>postp2_b(i,1)) & test(i,end)==1
        count_b=count_b+1
    elseif (postp1_b(i,1)<postp2_b(i,1)) & test(i,end)==2
        count_b=count_b+1
    end 
end
err_rate_b=1- count_b/100
%In question (c)
alph1= mean(diag(cov(training_c1(:,1:end-1))));
alph2= mean(diag(cov(training_c2(:,1:end-1))));
S1_c=alph1*eye(8)
S2_c=alph2*eye(8)
postp1_c = mvnpdf(test(:,1:end-1),mu_1,S1_c);
postp2_c = mvnpdf(test(:,1:end-1),mu_2,S2_c);
count_c=0
for i=1:100 
    if (postp1_c(i,1)>postp2_c(i,1)) & test(i,end)==1
        count_c=count_c+1
    elseif (postp1_c(i,1)<postp2_c(i,1)) & test(i,end)==2
        count_c=count_c+1
    end 
end
err_rate_c=1- count_c/100