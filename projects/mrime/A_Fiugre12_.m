clc
clear
close all
% 读取数据
load MRIME_Duibi_10D_3000xDim_51Runs_CEC2022
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
%     RANK=Results(j+5,:);
%     A_results_min_ave_std=real([A_results_min_ave_std;best;ave;Std;RANK]);
    A_results_min_ave_std=[A_results_min_ave_std;best;ave;Std];
    j=j+6;
end

j=1;
for i=1:func_num    
ave(i,:)=(Results(j+2,:)); 
j=j+6;
end
[p_mean,table_mean,stats_mean]=friedman(ave);    
close all    


A_rank_sta=zeros(size(rank_sum_RESULT,2),3);
AAAAA_Rank_tongji=[];

for i=1:size(rank_sum_RESULT,2)
    for j=1:length(F)
               if rank_sum_RESULT(j,i)<0.05
            if RANK_result (j,1)<RANK_result (j,i+1)   %比ARO好的
              A_rank_sta(i,1)=A_rank_sta(i,1)+1;
            else
              A_rank_sta(i,3)=A_rank_sta(i,3)+1;
            end
        else
            A_rank_sta(i,2)=A_rank_sta(i,2)+1;  %比ARO差的
        end       
    end
    AA=A_rank_sta(i,1);
    AB=A_rank_sta(i,2);
    AC=A_rank_sta(i,3);
    AAAAAAA= [num2str(AA),'/' num2str(AB),'/' num2str(AC)];
   % AAAAA_Rank_tongji=char(AAAAA_Rank_tongji,AAAAAAA);
     AAAAA_Rank_tongji=strvcat(AAAAA_Rank_tongji,AAAAAAA);
end
% 读取数据
% 自变量
X = 1:9;
% 因变量
A =A_rank_sta;


%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 13;
figureHeight =13;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [5 10 figureWidth figureHeight]); % define the new figure dimensions
hold on


AA=slanCL(824,1:12);
% CData=[111,173,72;92,154,215;255,192,1;69,103,42;36,94,144]./255;
C = AA;

GO = barh(X,A',0.8,'stacked','EdgeColor','k');


% 赋色
GO(1).FaceColor = C(4,:);
GO(2).FaceColor = C(5,:);
GO(3).FaceColor = C(7,:);
% GO(4).FaceColor = C(4,:);
% GO(5).FaceColor = C(5,:);
% GO(6).FaceColor = C(6,:);

% 坐标轴美化
set(gca, 'Box', 'off', ...                                       % 边框
        'XGrid', 'off', 'YGrid', 'off', ...                      % 网格
        'TickDir', 'out', 'TickLength', [.01 .01], ...           % 刻度
        'XMinorTick', 'off', 'YMinorTick', 'off', ...            % 小刻度
        'XColor', [.1 .1 .1],  'YColor',[.1 .1 .1],...           % 坐标轴颜色
        'XTick',0:2:size(rank_sum_RESULT,1)+2,...                                      % 刻度位置、间隔、范围
        'YTick',1:9,...                                     
        'Xlim' ,[0 size(rank_sum_RESULT,1)+1],...
        'Ylim' , [0.2 9.6], ...
        'Yticklabel',{'MRIME-CD vs. RIME' 'MRIME-CD vs. EO' 'MRIME-CD vs. SAO' 'MRIME-CD vs. ACGRIME' 'MRIME-CD vs. IRIME' 'MRIME-CD vs. TERIME' 'MRIME-CD vs. EOSMA' ...
        'MRIME-CD vs. RDGMVO','MRIME-CD vs. MTVSCA'},...    % Y坐标轴刻度标签
        'Xticklabel',{[0:2:size(rank_sum_RESULT,1)+2]})                                % X坐标轴刻度标签
    
    
% 标签及Legend 设置   
hLegend =legend([GO(1),GO(2),GO(3)], ...
                'win','equal','lose', ...
                 'Location','southeast','Orientation','horizontal','FontSize',15);
hLegend.ItemTokenSize = [10 10];
% 添加图例
%        AAA=['CEC 2022' '  (D=' num2str(variables_no) ')'];
%         title(AAA)
% lgd1=legend(hBar1,'win','equal','lose','FontSize',13);
hLegend.FontName='Arial';
% 设置图例位置
hLegend.Location='southoutside';
% 设置图例方形大小
hLegend.ItemTokenSize=[10,10];
% 关闭框
hLegend.Box='off';
legend('boxoff');


%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[5 10 figW figH]);
Function_name=['CEC2019 对比柱状图' ' Dim=' num2str(dim)]; 
print(figureHandle,[Function_name,'.tiff'],'-r600','-dtiff');