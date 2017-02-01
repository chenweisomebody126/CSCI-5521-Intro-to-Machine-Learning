function [z,w,v] = mlptrain(train_data,val_data,d,m,k,c,l,struct)
%Let d= size of imput,d=64
%Let m= number of hidden units except bias unit
%Let k= size of output,k=10
%Let c be the size of image,c=8 
%Let l be the size of sliding window?e.g. l=2, then sliding window is 2*2 
% Let strcut==3 represent non-structured MLP
% Let struct==1 represent structured MLP with overlapping
% Let struct==2 represent structured MLP without overlapping
traindata=load(train_data);
valdata=load(val_data);
if struct==3
    [z,w,v] = mlp_ondata(traindata,d,m,k);
    [~,train_err_rate]=err_rate_ondata(traindata,w,v);
    [~,val_err_rate]=err_rate_ondata(valdata,w,v);
elseif struct==1
    train_newX = overlap_structure_ondata(traindata,c,l);
    [z,w,v] = mlp_ondata(train_newX,size(train_newX,2)-1,(c-l+1)^2,k);
    [~, train_err_rate]=err_rate_ondata(train_newX,w,v);
    val_newX = overlap_structure_ondata(valdata,c,l);
    [~, val_err_rate]=err_rate_ondata(val_newX,w,v);
elseif struct==2
    train_newX = nonoverlap_structure_ondata(traindata,c,l);
    [z,w,v] = mlp_ondata(train_newX,size(train_newX,2)-1,(c/l)^2,k);
    [~,train_err_rate]=err_rate_ondata(train_newX,w,v);
    val_newX =nonoverlap_structure_ondata(valdata,c,l);
    [~,val_err_rate]=err_rate_ondata(val_newX,w,v);    
else
    return
end
 fprintf('train_err_rate= %d\n',train_err_rate);
 fprintf('val_err_rate= %d\n',val_err_rate);
end

