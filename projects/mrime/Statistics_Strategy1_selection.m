clc
clear
close all

%     load MSAO_EDA_Strategy1_Selection_100D_10000xDim_51Runs_CEC2017.mat
    load MSAO_EDA_Strategy1_Selection_10D_10000xDim_51Runs_CEC2022.mat
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


A_rank_sta=zeros(size(rank_sum_RESULT,2)+1,3);
AAAAA_Rank_tongji=[];

for i=1:size(rank_sum_RESULT,2)
    for j=1:length(F)
               if rank_sum_RESULT(j,i)<0.05
            if RANK_result (j,1)>RANK_result (j,i+1)   %比ARO好的
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
A_rank_sta(size(rank_sum_RESULT,2)+1,:)=sum(A_rank_sta(1:end,:),1);



%迭代图
%迭代图
MarkerL = {'v','o','^','s','d','v','o','^','s','d','v','o','^','s','d'};
AA=slanCL(1808,1:12);
 mycolor  =AA;
for i=1:func_num
 %% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 10;
figureHeight =8;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [5 10 figureWidth figureHeight]); % define the new figure dimensions
hold on

 curve_plot=curve_plot_total(:,:,i);
CNT=10;

Function_name=['F' num2str(F(i))];     
k=round(linspace(1,max_iter,CNT)); %随机选CNT个点
% 注意：如果收敛曲线画出来的点很少，随机点很稀疏，说明点取少了，这时应增加取点的数量，100、200、300等，逐渐增加
% 相反，如果收敛曲线上的随机点非常密集，说明点取多了，此时要减少取点数量
if i~=2&i~=10&i~=11&i~=12&i~=13&i~=14&i~=17&i~=18&i~=29
iter=1:1:max_iter;
    plot(iter(k),curve_plot(1,k),'LineStyle','-','Marker',MarkerL{1},'LineWidth',2,'color',mycolor(1,:));
    hold on
    plot(iter(k),curve_plot(2,k)+20,'LineStyle','-','Marker',MarkerL{2},'LineWidth',1.5,'color',mycolor(2,:));
    hold on
    plot(iter(k),curve_plot(3,k)+20,'LineStyle','-','Marker',MarkerL{3},'LineWidth',1.5,'color',mycolor(3,:));
    hold on
    plot(iter(k),curve_plot(4,k)+20,'LineStyle','-','Marker',MarkerL{4},'LineWidth',1.5,'color',mycolor(4,:));
    hold on
    plot(iter(k),curve_plot(5,k)+20,'LineStyle','-','Marker',MarkerL{5},'LineWidth',1.5,'color',mycolor(5,:));
    hold on
    plot(iter(k),curve_plot(6,k)+20,'LineStyle','-','Marker',MarkerL{6},'LineWidth',1.5,'color',mycolor(6,:));
    hold on
        plot(iter(k),curve_plot(7,k)+20,'LineStyle','-','Marker',MarkerL{7},'LineWidth',1.5,'color',mycolor(7,:));
    hold on
        plot(iter(k),curve_plot(8,k)+20,'LineStyle','-','Marker',MarkerL{8},'LineWidth',1.5,'color',mycolor(8,:));
    hold on
        plot(iter(k),curve_plot(9,k)+20,'LineStyle','-','Marker',MarkerL{9},'LineWidth',1.5,'color',mycolor(9,:));
    hold on
        plot(iter(k),curve_plot(10,k)+20,'LineStyle','-','Marker',MarkerL{10},'LineWidth',1.5,'color',mycolor(10,:));
    hold on
        plot(iter(k),curve_plot(11,k)+20,'LineStyle','-','Marker',MarkerL{11},'LineWidth',1.5,'color',mycolor(11,:));
    hold on
        plot(iter(k),curve_plot(12,k)+20,'LineStyle','-','Marker',MarkerL{12},'LineWidth',1.5,'color',mycolor(12,:));
    hold on

else
iter=1:1:max_iter;
    semilogy(iter(k),curve_plot(1,k),'LineStyle','-','Marker',MarkerL{1},'LineWidth',2,'color',mycolor(1,:));
    hold on
    semilogy(iter(k),curve_plot(2,k)+20,'LineStyle','-','Marker',MarkerL{2},'LineWidth',1.5,'color',mycolor(2,:));
    hold on
    semilogy(iter(k),curve_plot(3,k)+20,'LineStyle','-','Marker',MarkerL{3},'LineWidth',1.5,'color',mycolor(3,:));
    hold on
    semilogy(iter(k),curve_plot(4,k)+20,'LineStyle','-','Marker',MarkerL{4},'LineWidth',1.5,'color',mycolor(4,:));
    hold on
    semilogy(iter(k),curve_plot(5,k)+20,'LineStyle','-','Marker',MarkerL{5},'LineWidth',1.5,'color',mycolor(5,:));
    hold on
    semilogy(iter(k),curve_plot(6,k)+20,'LineStyle','-','Marker',MarkerL{6},'LineWidth',1.5,'color',mycolor(6,:));
    hold on
        semilogy(iter(k),curve_plot(7,k)+20,'LineStyle','-','Marker',MarkerL{7},'LineWidth',1.5,'color',mycolor(7,:));
    hold on
        semilogy(iter(k),curve_plot(8,k)+20,'LineStyle','-','Marker',MarkerL{8},'LineWidth',1.5,'color',mycolor(8,:));
    hold on
        semilogy(iter(k),curve_plot(9,k)+20,'LineStyle','-','Marker',MarkerL{9},'LineWidth',1.5,'color',mycolor(9,:));
    hold on
        semilogy(iter(k),curve_plot(10,k)+20,'LineStyle','-','Marker',MarkerL{10},'LineWidth',1.5,'color',mycolor(10,:));
    hold on
        plot(iter(k),curve_plot(11,k)+20,'LineStyle','-','Marker',MarkerL{11},'LineWidth',1.5,'color',mycolor(11,:));
    hold on
        semilogy(iter(k),curve_plot(12,k)+20,'LineStyle','-','Marker',MarkerL{12},'LineWidth',1.5,'color',mycolor(12,:));
    hold on

    
% set(gca,'yscale','log');    %修改x轴为对数
% semilogy(iter(k),curve_plot(1,k),'k-o','linewidth',2);
%     hold on
%     semilogy(iter(k),curve_plot(2,k),'c-v','linewidth',2);
%     hold on
%     semilogy(iter(k),curve_plot(3,k),'b-','linewidth',1);
%     hold on
%     semilogy(iter(k),curve_plot(4,k),'m-','linewidth',1);
%     hold on
%     semilogy(iter(k),curve_plot(5,k),'g-','linewidth',1);
%     hold on
%     semilogy(iter(k),curve_plot(6,k),'r-','linewidth',1);
end

       AAA=['CEC-2017 F' num2str(F(i))  ' D=' num2str(variables_no)];
        title(AAA)
grid on;
xlabel('Iterations');
ylabel('Mean Fitness');
box on
% % if i==29
legend('MICFOA','CFOA', 'PEOA','TLBO','COA','ARO','EDO', 'YDSE','LSHADE','JADE','IDE-EDA','APSM-jSO')
% end

% 显示图像
grid on;
% 字体和字号
set(gca, 'FontName', 'Helvetica')
% set([hXLabel, hYLabel, hLegend], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 13)
% set([hXLabel, hYLabel, hLegend], 'FontSize', 11)
%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[5 10 figW figH]);
fileout = 'test';
Function_name=['CEC2017 F' num2str(F(i)) '迭代图' 'dim=' num2str(variables_no)]; 
print(figureHandle,[Function_name,'.tiff'],'-r600','-dtiff');
% print(figureHandle,[Function_name,'.fig'],'-r600');


end


close all

%这个是画箱式图
figure()
AA=slanCL(1808,1:12);
 mycolor  =AA;
for j=1:29
 %% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 10;
figureHeight =8;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [5 10 figureWidth figureHeight]); % define the new figure dimensions
hold on
box_plot=total(:,:,j);
 func_num=j;
 
%这个是画箱式图
%         mycolor = [0.2 0.2 0.6;... %深蓝色
%        1 0 0;... %红色
%   0.5 0.5 0.2;...%深黄色
%         0.9 0.2 0.6;... %红紫色
%         0.1 0.8 0.4;...%蓝绿色
%        0.4 0.4 0.7;...%灰蓝色
%         0.4,0.8,0.1;...%绿黄色
%         0.9,0.6,0.2;...%橙黄色
%         0.7,0.7,0.7;%浅灰色
%          0.8,0.1,0.4;];  %设置一个颜色库
        %% 开始绘图
        %参数依次为数据矩阵、颜色设置、标记符
        box_figure = boxplot(box_plot','color',[0 0 1],'Symbol','o');
        %设置线宽
        set(box_figure,'Linewidth',1.2);
        boxobj = findobj(gca,'Tag','Box');
        for i = 1:12   %因为总共有4个算法，这里根据自身实际情况更改
            patch(get(boxobj(i),'XData'),get(boxobj(i),'YData'),mycolor(i,:),'FaceAlpha',0.5,...
                'LineWidth',0.7);
        end

        set(gca,'XTickLabel',{'MICFOA','CFOA', 'PEOA','TLBO','COA','ARO','EDO', 'YDSE','LSHADE','JADE','IDE-EDA','APSM-jSO'});
        set(gca,'XTickLabelRotation',45); 
        AAA=['CEC-2017 F' num2str(F(j))  ' D=' num2str(variables_no)];
        title(AAA)

      % 显示图像
grid on;
% 字体和字号
set(gca, 'FontName', 'Helvetica')
% set([hXLabel, hYLabel, hLegend], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 13)
% set([hXLabel, hYLabel, hLegend], 'FontSize', 11)
%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[5 10 figW figH]);
fileout = 'test';
Function_name=['2017 F' num2str(F(j)) '箱式图' num2str(F(j))  'dim=' num2str(variables_no)]; 
print(figureHandle,[Function_name,'.tiff'],'-r600','-dtiff');
% print(figureHandle,[Function_name,'.fig'],'-r600');
  
        
end
close all

    