% to convert intial csv files
%load all data and save only the desired columns
clear all;
close all;
A=dir('IITH*.csv');
[directory,T_des]=TIRTL_desired(A);
% C=colormap(jet(size(directory,1)));
T_all=table();
for ii=1:length(T_des)
    T_all=[T_all;T_des{ii}];
end
% Converting the data in to standard datetime format
T_all.datetime=dateshift(T_all.date,'start','day') + timeofday(T_all.time);
T_all.datetime.Format='yyyy/MM/dd HH:mm:ss';
cf_t=duration(0,3,00); % manually observed the lag/lead in ITRTL(time(hh,mm,ss))
T_all.datetime = T_all.datetime-duration(cf_t);
T_all.date=[];
T_all.time=[];
save('T_all','T_all');

%%
load T_all.mat
T_all=T_all(~isnat(T_all.datetime),:);
% T_allbig=T_all(~ismember(T_all.(3),[1,3]),:);
T=[dateshift(T_all.datetime(1),'start','day'):seconds(1):dateshift(T_all.datetime(end),'end','day')];
% Tbig=T(~ismember(T_all.(3),[1,3]));
[TIRTL_hist_allclas,edges]=(histcounts(T_all.datetime,T));
% [TIRTL_hist_bigclas,edges]=(histcounts(T_allbig.datetime,T));
TIRTL_hist_allclas=TIRTL_hist_allclas';
[A,edg,binn]=histcounts(T_all.datetime,edges);
TIRTL_avgspd=accumarray(binn,T_all.average_speedkph,[length(edges)-1 1],@mean);
TIRTL_avgspd(TIRTL_avgspd==0)=nan;
clear T_all;

% TIRTL_hist_bigclas=TIRTL_hist_bigclas';
% [A,edg,binn]=histcounts(T_allbig.datetime,edges);
% TIRTLbigcls_avgspd=accumarray(binn,T_allbig.average_speedkph,[length(edges)-1 1],@mean);
% TIRTLbigcls_avgspd(TIRTL_avgspd==0)=nan;
T=T(1:end-1)';
TIRTL_hist_allclasmat=table(T,TIRTL_hist_allclas);
% time_ind_TIRTL=(ST_time+duration([00 00 blck(jj)]):duration([00 00 blck(jj)]): ED_time)';
writetable(TIRTL_hist_allclasmat,'TIRTL_hist_allclas.csv')
c
% save('T_all','T_all');
save('TIRTL_hist_allclas','T','TIRTL_hist_allclas','TIRTL_avgspd');
% save('TIRTL_hist_bigclas','Tbig','TIRTL_hist_bigclas','TIRTLbigcls_avgspd');
%%
blck=[10 60 300 600 1200];
for jj=1:length(blck)

    TT=(T(1):seconds(blck(jj)):T(end))';
    T_hist=(nansum(reshape(TIRTL_hist_allclas,[blck(jj) length(TIRTL_hist_allclas)/blck(jj)])))';
    Tab=table(TT,T_hist);
    eval(['TIRTL_hist_allclas_' num2str(blck(jj)) 's=Tab;']);
% writetable(TIRTL_hist_allclas_60s,'TIRTL_hist_allclas.csv')
    eval(['save(''TIRTL_hist_allclas_' num2str(blck(jj)) 's'',''TIRTL_hist_allclas_' num2str(blck(jj)) 's'')']);

end

writetable(TIRTL_hist_allclas_60s,'TIRTL_hist_allclas_60s.csv')


    %     Tab2=table;
    %     Tab2.datetime=datetimemat;
    %     Tab2(:,2:9)=array2table(mat2);
    %     Tab2.Properties.VariableNames(2:9)=[clmnm];
%     eval(['bigmat_mn_' num2str(blck(jj)) 's=Tab1;']);
    %     eval(['bigmat_med_' num2str(blck(jj)) 's_mvmd3=Tab11;']);
    %     eval(['bigmat_mvmd_' num2str(blck(jj)) 's=Tab2;']);
%     eval(['save(''bigmat_mn_' num2str(blck(jj)) 's'',''bigmat_mn_' num2str(blck(jj)) 's'')']);
% end