function [ test_z, error_rate] = err_rate_ondata( testdata, w,v)

test_t=size(testdata,1);
test_r=testdata(:,end);
test_x=[ones(test_t,1) testdata(:,1:end-1)];
test_z=1./(1+exp(-test_x*transpose(w)));
test_z_plus=[ones(test_t,1) test_z];
test_o=test_z_plus*transpose(v);


label=zeros(test_t,1);
for t=1:test_t
    test_y(t,:)=exp(test_o(t,:))/sum(exp(test_o(t,:)));
    [~,idx]=max(test_y(t,:));
    label(t,1)=idx-1;
end
error_rate=sum(label~=test_r)/test_t;
end