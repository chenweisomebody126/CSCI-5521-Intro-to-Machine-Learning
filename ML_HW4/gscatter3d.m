function [] = gscatter3d( datafile )

color=[0 1 0; 1 0 1; 0 1 1; 1 0 0; .2 .6 1; 1 1 1; 1 .6 .2; 0 0 1; 1 .2 .6; .2 1 .6];

[z,~,~ ] = mlp(datafile,64,3,10);
data=load(datafile);
figure
for k=0:9
    z_group= z(data(:,end)==k,:);
    scatter3(z_group(:,1),z_group(:,2),z_group(:,3),[],color(k+1,:));
    hold on;
end
for t=1:size(z,1)
    text(z(t,1),z(t,2),z(t,3),num2str(data(t,end)));
end

end

