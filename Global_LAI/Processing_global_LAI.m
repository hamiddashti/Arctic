%% This code was used for comparing the global mean LAI between three datasets
% GLOBMap LAI; GLASS LAI and GIMMS3g LAI. 
% The GLOBMap and GLASS data were recieved from Wenping Yuan, 
% Increased atmospheric vapor pressure deficit reduces global vegetation growth
% GIMSS the one that Andy used in his global LAI paper. 

clc
clear all
cd('P:\nasa_above\global_lai\LAI')
%% 
globemap = load('GLOBMAP_Fill.mat');
glass = load('GLASS_Fill.mat');
GIMMS = load ('GIMMS.mat');

%tmp = ncread('GIMMS3g_LAI_Bimonthly_2000_2010_60min.nc','LAI');
%tmp2 = tmp(:,:,1);
globmap_data = globemap.data;
glass_data = glass.data;
modis_data = GIMMS.data;
modis_data = permute(modis_data,[2,3,1]);
modis_data = modis_data*0.1;

%glob_mean = nanmean(globmap_data,3);
%glass_mean = nanmean(glass_data,3);

%glob_mean = globmap_data(:,:,720);
%glass_mean = glass_data(:,:,720);
%modis_mean = modis_data(:,:,264);



d1 = nanmean(globmap_data,3);
v1 = d1(:);
id1 = find(v1(:) > 0);
g1 = d1(id1);
x1=(linspace(0,5,length(g1)))';
pd1 = fitdist(g1,'Normal')
y1 = pdf(pd1,x1);


d2 = nanmean(glass_data,3);
v2 = d2(:);
id2 = find(v2(:) > 0);
g2 = d2(id2);
x2=(linspace(0,5,length(g2)))';
pd2 = fitdist(g2,'Normal')
y2 = pdf(pd2,x2);

d3 = nanmean(modis_data,3);
v3 = d3(:);
id3 = find(v3(:) > 0);
g3 = d3(id3);
x3=(linspace(0,5,length(g3)))';
pd3 = fitdist(g3,'Normal')
y3 = pdf(pd3,x3);


figure(1)
plot(x1,y1,'-.','LineWidth',2)
hold on
plot(x2,y2,'LineWidth',2)
hold on
plot(x3,y3,':','LineWidth',2)


xlabel("Leaf Area Index [m2/m2]",'fontsize',18)
ylabel("Normal PDF",'fontsize',18)
set(gcf,'color','w','position',[500,250,700,650])
set(gca,'FontSize',18,'fontname','times')
legend('GLOBMap LAI','GLASS LAI','GIMMS3g')
