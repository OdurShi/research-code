clc
clear
close all
    load MEARO_duibi_100D_500X30_30Runs_CEC2017
%       load MARO_duibi_10D_500X30_30Runs_CEC2022
%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth =30;
figureHeight = 6;
%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [6 12 figureWidth figureHeight]); % define the new figure dimensions
hold on

Data=RANK_result';
SHM=SHeatmap(Data,'Format','sq');
SHM=SHM.draw(); 

ax=gca;
SHM.setText();
%  ax.XTickLabel={'F1','F2','F3','F4','F5','F6','F7','F8',...
%     'F9','F10','F11','F12','F13','F14','F15','F16','F17','F18',...
%     'F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29'};
ax.XTickLabel={'F1','F2','F3','F4','F5','F6','F7','F8',...
    'F9','F10','F11','F12'};
 ax.YTickLabel={'MEARO','ARO','RIME','PSA','SMA','COA','DTSMA','AFDBARO','FDBARO','LCAHA'};
ax.FontSize=14;
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 12)
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
% set(figureHandle,'PaperUnits',figureUnits);
% set(figureHandle,'PaperPosition',[6 12 figW figH]);
% fileout = 'FTS策略对比结果';
% print(figureHandle,[fileout,'.tiff'],'-r600','-dtiff');
fileout = 'test';
Function_name=['2017' '热力图'  'Dim=' num2str(variables_no)]; 
print(figureHandle,[Function_name,'.tiff'],'-r600','-dtiff');