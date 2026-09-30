function [x] = Cubic
x(1)=0.3; %
na = 2.59 ;
for i=1:499 
% x(i+1) = na*x(i)*(1-x(i)^2);
x(i+1) = rand;
end
% figure
% grid on
% axis tight
% plot(x,'-*','markersize',6);
% title('Random','fontsize',14,'FontName','Times New Roman');
% xlabel('Iteration','fontsize',14,'FontName','Times New Roman')
% ylabel('Chaotic Sequences','fontsize',14,'FontName','Times New Roman')
% set(gca,'FontSize',14,'FontName','Times New Roman','LineWidth',2);
% figure
% histogram(x)
% grid on
% axis tight
% title('Random','fontsize',14,'FontName','Times New Roman');
% xlabel('Chaotic Sequences','fontsize',14,'FontName','Times New Roman')
% ylabel('Frequency Counts','fontsize',14,'FontName','Times New Roman')
% set(gca,'FontSize',14,'FontName','Times New Roman','LineWidth',2);
end
