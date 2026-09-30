clc
clear
close all

    load MRIME_Duibi_20D_3000xDim_51Runs_CEC2022
clear title
F=[1:func_num];
%%%%%%%%%统计 最优值、平均值、标准差、排名%%%%%%%%%%%%%%%
Results=RESULT;%[min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
j=1;
A_results_min_ave_std=[];
for i=1:func_num
    best=Results(j,:);
    ave=Results(j+2,:);
    Std=Results(j+1,:);
     RANK=Results(j+5,:);
     A_results_min_ave_std=real([A_results_min_ave_std;best;ave;Std;RANK]);
%     A_results_min_ave_std=[A_results_min_ave_std;best;ave;Std];
    j=j+6;
end

j=1;
for i=1:func_num    
ave(i,:)=(Results(j+2,:)); 
j=j+6;
end
[p_mean,table_mean,stats_mean]=friedman(ave);    
close all    


% A_rank_sta=zeros(size(rank_sum_RESULT,2)+1,3);
% AAAAA_Rank_tongji=[];
% 
% for i=1:size(rank_sum_RESULT,2)
%     for j=1:length(F)
%                if rank_sum_RESULT(j,i)<0.05
%             if RANK_result (j,1)<RANK_result (j,i+1)   %比ARO好的
%               A_rank_sta(i,1)=A_rank_sta(i,1)+1;
%             else
%               A_rank_sta(i,3)=A_rank_sta(i,3)+1;
%             end
%         else
%             A_rank_sta(i,2)=A_rank_sta(i,2)+1;  %比ARO差的
%         end       
%     end
%     AA=A_rank_sta(i,1);
%     AB=A_rank_sta(i,2);
%     AC=A_rank_sta(i,3);
%     AAAAAAA= [num2str(AA),'/' num2str(AB),'/' num2str(AC)];
%    % AAAAA_Rank_tongji=char(AAAAA_Rank_tongji,AAAAAAA);
%      AAAAA_Rank_tongji=strvcat(AAAAA_Rank_tongji,AAAAAAA);
% end
% A_rank_sta(size(rank_sum_RESULT,2)+1,:)=sum(A_rank_sta(1:end,:),1);


B_rank_sta=zeros(size(rank_sum_RESULT,2)+1,3);
BAAAA_Rank_tongji=[];

for i=1:size(rank_sum_RESULT,2)
    for j=1:length(F)
               if kruskalwallis_sum_RESULT(j,i)<0.05
            if RANK_result (j,1)<RANK_result (j,i+1)   %比ARO好的
              B_rank_sta(i,1)=B_rank_sta(i,1)+1;
            else
              B_rank_sta(i,3)=B_rank_sta(i,3)+1;
            end
        else
            B_rank_sta(i,2)=B_rank_sta(i,2)+1;  %比ARO差的
        end       
    end
    AA=B_rank_sta(i,1);
    AB=B_rank_sta(i,2);
    AC=B_rank_sta(i,3);
    AAAAAAA= [num2str(AA),'/' num2str(AB),'/' num2str(AC)];
   % AAAAA_Rank_tongji=char(AAAAA_Rank_tongji,AAAAAAA);
     BAAAA_Rank_tongji=strvcat(BAAAA_Rank_tongji,AAAAAAA);
end
B_rank_sta(size(rank_sum_RESULT,2)+1,:)=sum(B_rank_sta(1:end,:),1);


%迭代图
%迭代图
MarkerL = {'v','o','^','s','d','v','o','^','s','d','v','o','^','s','d'};
AAAAAA=mean(kruskalwallis_Rank_RESULT);

    