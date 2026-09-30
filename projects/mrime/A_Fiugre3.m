clc
clear
close all

A=[6.97 	4.79 	5.45 	5.38 	3.86 	3.66 	3.90 	2.00 
7.66 	2.97 	6.07 	6.86 	2.83 	2.90 	4.97 	1.76 
7.55 	3.14 	5.76 	6.83 	2.83 	3.66 	4.97 	1.28 
7.55 	3.14 	5.41 	7.24 	2.55 	3.93 	5.17 	1.00 
7.43 	3.51 	5.67 	6.58 	3.02 	3.53 	4.75 	1.51 							
6.92 	4.00 	5.92 	5.17 	4.17 	3.25 	4.25 	2.33 
7.50 	3.67 	6.08 	6.08 	3.00 	2.92 	5.08 	1.67 
7.21 	3.83 	6.00 	5.63 	3.58 	3.08 	4.67 	2.00 
];
%几种情况
NNNNN=8;
x=[1:8];
%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 25;
figureHeight = 14;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [3 3 figureWidth figureHeight]); % define the new figure dimensions
hold on
p = plot(x,A);
% hTitle = title('Line Plot');
% hXLabel = xlabel('XAxis');
hYLabel = ylabel('Friedman ranking of different test suite');

% 线条属性调整
MarkerL = {'o','^','s','d','v','o','^','s','d','v','o','^','s','d'};
AA=slanCL(824,1:12);
 mycolor  =AA;
%    mycolor = [0.2 0.2 0.6;... %深蓝色
%        1 0 0;... %红色
%   0.5 0.5 0.2;...%深黄色
%         0.9 0.2 0.6;... %红紫色
%         0.1 0.8 0.4;...%蓝绿色
%        0.4 0.4 0.7;...%灰蓝色
%         0.4,0.8,0.1;...%绿黄色
%         0.9,0.6,0.2;...%橙黄色
%         0.7,0.7,0.7;%浅灰色
%          0.8,0.1,0.4;];  %设置一个颜色库
% AA=slanCL(1808,1:12);
%  mycolor  =AA;
%       mycolor  = [0.3059    0.4745    0.6549
%     0.9490    0.5569    0.1686
%     0.8824    0.3412    0.3490
%     0.4627    0.7176    0.6980
%     0.3490    0.6314    0.3098
%     0.9294    0.7882    0.2824
%     0.6902    0.4784    0.6314
%     1.0000    0.6157    0.6549
%     0.6118    0.4588    0.3725
%     0.7294    0.6902    0.6745];
for i = 1:8
    set(p(i),'LineStyle','-','Marker',MarkerL{i},'LineWidth',2,'color',mycolor(i,:))
end
% 坐标区属性调整
set(gca, 'Box', 'off', ...                                % 边框
         'LineWidth', 1,...                               % 线宽
         'XGrid', 'off', 'YGrid', 'on', ...               % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...   % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...    % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])     % 坐标轴颜色
% 坐标轴刻度调整
set(gca, 'XTick', 1:1:NNNNN,  'YTick', 1:1:8,...            % 刻度位置、间隔
         'Xlim' ,[1 NNNNN],'Ylim' ,[1 8], ...                % 坐标轴范围
         'Xticklabel',{ 'CEC17(D=10)', 'CEC17(D=30)', 'CEC17(D=50)', 'CEC17(D=100)','Mean Ranking of CEC17', 'CEC22 (D=10)','CEC22 (D=20)','Mean Ranking of CEC22'},...                         % X坐标轴刻度标签
         'Yticklabel',{1:1:NNNNN})                          % Y坐标轴刻度标签
      set(gca,'XTickLabelRotation',20);
% Legend
hLegend = legend(p, ...
                 'RIME','RIME-G','RIME-A','RIME-S','RIME-GA','RIME-GS','RIME-AS','MRIME-CD',...
                 'Location', 'north');
             legend('NumColumns',6);
% Legend位置微调 
P = hLegend.Position;
hLegend.Position = P + [0.01 0.03 0 0];
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 12)
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[3 3 figW figH]);
fileout = '改进策略对比结果';
print(figureHandle,[fileout,'.tiff'],'-r600','-dtiff');