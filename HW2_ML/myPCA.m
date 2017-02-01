faces=load('faces.txt');
%Q3(1)
%implement the PCA and find the first two eigenvectors
cov_faces=cov(faces);
[V_faces, L_faces]=eig(cov_faces);
compo_1st=V_faces(:,end);
compo_2nd=V_faces(:,end-1);
%plot the eigenvectors seperately
subplot(1,2,1);
imagesc(reshape(compo_1st,60,64));
title('1st component');
subplot(1,2,2);
imagesc(reshape(compo_2nd,60,64));
title('2nd component')
%Q3(2)
V_faces_descend=fliplr(V_faces);
new_d=[10,50,100];
for i=1:length(new_d)
  V_1stRow=V_faces_descend(:,1:new_d(i));
  centered_faces=faces(1,:)-mean(faces,1);
  project_centered_faces=centered_faces(1,:)*V_1stRow;
  backproject_centered_faces=project_centered_faces*transpose(V_1stRow);
  backproject_faces= backproject_centered_faces+mean(faces,1);
  subplot(1,3,i);
  imagesc(reshape(backproject_faces,60,64));
end