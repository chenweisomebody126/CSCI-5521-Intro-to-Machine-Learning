
%(a)(b)
[ h,M,Q] = EMG('stadium.bmp',4);
[ h,M,Q] = EMG('stadium.bmp',8);
[ h,M,Q] = EMG('stadium.bmp',12);
%(c)
%First uses built-in kMeans to display compressed image with k=7
[ima,cmap]= imread('goldy.bmp');
img_rgb =ind2rgb(ima,cmap);
img_double=im2double(img_rgb);
goldy=reshape(img_double, [],3);
[idx, M]=kmeans(goldy,7);
N=size(goldy,1);
Compress=zeros(N,3);
for j=1:N
    Compress(j,:)=M(idx(j),:);
end
Compress=reshape(Compress, size(img_rgb,1),size(img_rgb,2),3);        
figure
imagesc(Compress);
%Implement the EM step
[ h,M,Q] = EMG('goldy.bmp',7);