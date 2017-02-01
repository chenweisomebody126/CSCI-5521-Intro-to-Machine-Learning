function  boundary(x,t, Kvals  )
%x is the test sample matrix without labels
%t is the test labels 
%Kvals is a vector containing all values of k for kNN
[Xv Yv] = meshgrid(min(x(:,1)):(0.01):max(x(:,1)),min(x(:,2)):(0.01):max(x(:,2)));
N=size(x,1);


%show how many labels in test dataset using unique func
tv = unique(t);
for kv = 1:length(Kvals)
    K = Kvals(kv);
    
    classes = zeros(size(Xv));
    
    for i = 1:length(Xv(:))
        
        this = [Xv(i) Yv(i)];
        dists = sum((x - repmat(this,N,1)).^2,2);
        [d I] = sort(dists,'ascend');
        [a,b] = hist(t(I(1:K)));
        pos = find(a==max(a));
        if length(pos)>1
            order = randperm(length(pos));
            pos = pos(order(1));
        end
        classes(i) = b(pos);
    end
    figure(K); hold off
    for i = 1:length(tv)
        pos = find(t==tv(i));
        %pos_train=find(train_lab==t(i))
        scatter(x(pos,1),x(pos,2))
        text(x(pos,1),x(pos,2),num2str(t(pos,1)));
        hold on
    end
    contour(Xv,Yv,classes,[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],'k')
    ti = sprintf('K = %g',K);
    title(ti);

end

