clear all;
close all;
load TIRTL_hist_allclas_300s.mat;
tirtl=TIRTL_hist_allclas_300s;
dyws=reshape(tirtl.T_hist,[],height(tirtl)/(12*24));
cs=mean(dyws);
dyws=dyws(cs,:);
datacln=reshape(dyws,[],1);
plot(datacln)