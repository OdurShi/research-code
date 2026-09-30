clc
clear
close all
% groupBarDemo.m
% figure('Position',[500,200,850,550])
clear title

YData1=[290.1954023	164.3786342	268.4347532	248.3853955	162.7525355	151.7085869	223.7491548	126.3955375
310.0371873	140.7214334	273.9682218	287.4279919	131.3840433	138.2745098	241.3711968	112.8154158
309.1473969	145.7721433	263.469236	291.9939148	132.4638269	150.260311	242.959432	99.93373901
313.9749831	159.851927	260.1467208	307	140.6896552	165.2109533	256.1764706	32.94929006
296.3235294	153.6911765	278	244.7924837	175.9771242	139.9526144	223.1715686	124.0915033
310.8529412	145.6797386	278.0245098	258.3349673	154.4509804	143.4787582	228.872549	116.3055556
]';
% 原图片配色(matlab要求配色范围0-1因此要除以255)
% AAAAAA=slanCL(1808)

CData=slanCL(824,2:8);
 %% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 25;
figureHeight = 8;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [3 3 figureWidth figureHeight]); % define the new figure dimensions
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

%         title(AAA)
ax1.XTickLabel={'RIME','RIME-G','RIME-A','RIME-S','RIME-GA','RIME-GS','RIME-AS','MRIME-CD'};
% 修改坐标区域字体
ax1.FontName='Arial';
% ax1.FontWeight='bold';
ax1.FontSize=12;
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
lgd1=legend(hBar1,'D=10','D=30','D=50','D=100','FontSize',14);
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
set(figureHandle,'PaperPosition',[3 3 figW figH]);
fileout = 'test';
Function_name=['改进策略KW检验' ]; 
print(figureHandle,[Function_name,'.tiff'],'-r600','-dtiff');


