%% to convert intial csv files 
% load all data and save only the desired columns
function [B,Vt_org]=TIRTL_desired(A)
B=A;
for ii=1:length(A)
    S=A(ii).name;
    str=char(S);
    str=str(1:21);
    str=append(string(strrep(str,str(1:13),'T')),'.csv');
    V=importTIRTL(S,[2,inf]);
    Vt_org{ii}=V;
%     writetable(V,str);
    B(ii).name=str;
end
end