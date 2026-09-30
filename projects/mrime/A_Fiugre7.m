clc
clear
close all
% groupBarDemo.m
% figure('Position',[500,200,850,550])
    load MRIME_Duibi_100D_3000xDim_51Runs_CEC2017
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


A_rank_sta=zeros(size(rank_sum_RESULT,2)+1,3);
AAAAA_Rank_tongji=[];

for i=1:size(rank_sum_RESULT,2)
    for j=1:length(F)
               if rank_sum_RESULT(j,i)<0.05
            if RANK_result (j,1)<RANK_result (j,i+1)   %
              A_rank_sta(i,1)=A_rank_sta(i,1)+1;
            else
              A_rank_sta(i,3)=A_rank_sta(i,3)+1;
            end
        else
            A_rank_sta(i,2)=A_rank_sta(i,2)+1;  %比
        end       
    end
    AA=A_rank_sta(i,1);
    AB=A_rank_sta(i,2);
    AC=A_rank_sta(i,3);
    AAAAAAA= [num2str(AA),'/' num2str(AB),'/' num2str(AC)];
   % AAAAA_Rank_tongji=char(AAAAA_Rank_tongji,AAAAAAA);
     AAAAA_Rank_tongji=strvcat(AAAAA_Rank_tongji,AAAAAAA);
end
A_rank_sta(size(rank_sum_RESULT,2)+1,:)=sum(A_rank_sta(1:end,:),1);
% 自行构造的数据
% YData=[0.7772    1.0363    0.7772    1.0363    2.7202;
%        3.4974    5.3109    4.2746    4.0155   12.9534;
%        3.4974    3.7565    3.1088    3.8860   14.8964;
%        1.9430    2.3316    2.3316    2.7202    2.7202;
%       17.2280   19.9482   17.0984   19.0415   22.6684;
%        0.2591    0.3886    0.3886    0.3886    2.5907];
YData1=A_rank_sta(1:end-1,:);
% 原图片配色(matlab要求配色范围0-1因此要除以255)
% AAAAAA=slanCL(1808)

CData=slanCL(824,[3,5,7]);
 %% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 20;
figureHeight = 8;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [5 10 figureWidth figureHeight]); % define the new figure dimensions
hold on


% -------------------------------------------------------------------------
% 上子图
ax1=subplot(1,1,1);hold on
hBar1=bar(YData1);
% -------------------------------------------------------------------------
% 分组柱状图修饰
for i=1:length(hBar1)
    hBar1(i).EdgeColor='none';      % 轮廓无色
    hBar1(i).FaceColor=CData(i,:);  % 设置颜色
  
end
% -------------------------------------------------------------------------
% 修改X轴标签文本
ax1.XTick=1:size(YData1,1);
  AAA=['CEC 2017' '(' num2str(dim) 'D)'];
%         title(AAA)
ax1.XTickLabel={'RIME','EO','SAO','ACGRIME','IRIME', 'TERIME','EOSMA','RDGMVO','MTVSCA'};
% 修改坐标区域字体
ax1.FontName='Arial';
% ax1.FontWeight='bold';
ax1.FontSize=10;
% 添加网格并修饰
ax1.XGrid='on';
ax1.GridAlpha=.2;
% 框修饰
ax1.Box='on';
ax1.LineWidth=1.0;
% 刻度长度设置为0
ax1.TickLength=[0,0];
 set(gca,'XTickLabelRotation',0);
% set(gca,  'YTick', 1:1:num)            % 刻度位置、间隔

% set(gca, 'Box', 'off', ...                                         % 边框
%          'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
%          'TickDir', 'out', 'TickLength', [.005 .005], ...          % 刻度
%          'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
%          'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])              % 坐标轴颜色
% -------------------------------------------------------------------------
% 绘制辅助线
XV=(1:size(YData1,1)-1)+.5;
for i=1:length(XV)
    xline(ax1,XV(i),'LineWidth',1.4,'LineStyle','--','Color',[0,0,0]);
end
% -------------------------------------------------------------------------
% 添加图例
lgd1=legend(hBar1,'win','equal','lose','FontSize',14);
ax1.FontName='Arial';
% 设置图例位置
lgd1.Location='southoutside';
% 设置图例横向排列
lgd1.NumColumns=length(hBar1);
% 设置图例方形大小
% lgd1.ItemTokenSize=[8,8];
% 关闭框
lgd1.Box='off';
% -------------------------------------------------------------------------
%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[5 10 figW figH]);
fileout = 'test';
Function_name=['CEC2017' '对比算法柱状图' 'dim=' num2str(dim)]; 
print(figureHandle,[Function_name,'.tiff'],'-r600','-dtiff');


