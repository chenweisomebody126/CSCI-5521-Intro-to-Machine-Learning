function [newX] =nonoverlap_structure_ondata( traindata,c,l )

%Let c be the size of image,c=8 
%Let l be the size of sliding window?e.g. l=2, then sliding window is 2*2 
X=traindata(:,1:end-1);
T=size(X,1);

imageX=zeros(c,c,T);
%Let Num_step be the number of steps sliding window can move along one direction;
%e.g. Num_step=7
%Then sliding window can move Num_step^2 steps in total 
%and each step generates Num_entr= l^2 entries in one window 
Num_step=c/l;
Num_entr=l^2;
newX=zeros(T,(Num_step^2)*Num_entr);
for t=1:T
    %retrieve one image from orginal data matrix
    imageX(:,:,t)=reshape(X(t,:),c,c);
    %Consider the upper left corner of sliding window move from (1,1) down
    %to(2,1),(3,1) to (Num_step,1), and then to (1,2),(2,2)...until lower
    %right corner (Num_step, Num_step)
    for image_col=1:l:c-l+1
        for image_row=1:l:c-l+1
     %We retrieve entries through thier index on image instead of matrix
     %manipulating. We save the row index and coloumn index of entries of interest  
            image_row_idx=repmat(image_row:image_row+l-1,1,l);
            image_col_idx=reshape(repmat(image_col:image_col+l-1,l,1),1,[]);
            image_idx=sub2ind(size(imageX(:,:,t)),image_row_idx, image_col_idx);
            %We need to first retrieve this entire image matrix as imageplane 
            %and use index to retrieve entries from imageplane 
            imagePlane=imageX(:,:,t);
            %Note you also need to allocate the retrieved entries into newX
            %through the window_inx,e.g. 1st window of 2*2 in 3nd record should be in position (3,1:4)
            window_idx=sub2ind([Num_step, Num_step], (image_row-1)/l+1, (image_col-1)/l+1);
            newX(t,window_idx*Num_entr-Num_entr+1:window_idx*Num_entr)=imagePlane(image_idx);
        end
    end
end

newX=[newX traindata(:,end)];


end