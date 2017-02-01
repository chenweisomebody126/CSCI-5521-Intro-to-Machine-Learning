function [test_z] = mlptest(test_data,w,v,m,c,l,struct)
%Let d= size of imput
%Let m= number of hidden units except bias unit
%Let k= size of output
%Let c be the size of image,c=8 
%Let l be the size of sliding window?e.g. l=2, then sliding window is 2*2 
% Let strcut==3 represent non-structured MLP
% Let struct==1 represent structured MLP with overlapping
% Let struct==2 represent structured MLP without overlapping
testdata=load(test_data);
if struct==3
    [test_z,test_err_rate]=err_rate_ondata(testdata,w,v);
    
elseif struct==1
    test_newX = overlap_structure_ondata(testdata,c,l);
    [test_z,test_err_rate]=err_rate_ondata(test_newX,w,v);
    
elseif struct==2
    test_newX = nonoverlap_structure_ondata(testdata,c,l);
    [test_z,test_err_rate]=err_rate_ondata(test_newX,w,v);
 
else
    return
end
 fprintf('test_err_rate= %d\n',test_err_rate);
end
