

%%
clear
clc
close all
Rand_Seeds=load('input_data\Rand_Seeds.txt');
addpath(genpath(pwd));
addpath(genpath(pwd));

RANK_result=[];%统计排名
run =51;
box_pp = 0;  %可选1，或者其他。当等于1，绘制箱型图，否则不绘制
RESULT=[];   %统计标准差，平均值，最优值等结果
rank_sum_RESULT=[];  %统计秩和检验结果
kruskalwallis_sum_RESULT=[];  %统计秩和检验结果
kruskalwallis_Rank_RESULT=[];
draw_IF = 0;  %可选1，或者其他。当等于1，分别绘制图，否则不绘制
draw_all=0;  %可选1，或者其他。当等于1，绘制一起的图，否则不绘制
F = [1,3:30];%
dim =100; % 可选 10, 30, 50,100
% pop_size=500;   %种群数目
% max_iter=300;   %迭代次数


max_iter=3000*dim;
FesMax=max_iter;
pop_size=ceil(15*dim^(2/3));;   %种群数目
disp(['正在统计的是维度为',num2str(dim),'的CEC2017函数集'])

%        MRIME_Duibi_100D_3000xDim_51Runs_CEC2017 = 'test.mat';
%          save MRIME_Duibi_100D_3000xDim_51Runs_CEC2017
for func_num =1:length(F)   
    % Display the comprehensive results
    disp(['F',num2str(F(func_num)),'函数计算结果：'])
    num=F(func_num);
    [lower_bound,upper_bound,dim,fobj]=Get_Functions_cec2017(num,dim);  % [lb,ub,D,y]：下界、上界、维度、目标函数表达式
    resu = [];  %统计标准差，平均值，最优值等结果
    
    rank_sum_resu = [];   %统计秩和检验结果
    kruskalwallis_sum_resu = [];   %统计秩和检验结果
    box_plot = [];  %统计箱型图结果
      curve_plot=[];  %统计迭代图结果

    parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=MRIME_RL_Cov(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
             final_curve(nrun,:)=iter(1:max_iter);
        z1(nrun) =  final;
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
      curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    disp(['MPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);
   
   
    parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=RIME(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
             final_curve(nrun,:)=iter(1:max_iter);
        z2(nrun) =  final;
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA_1：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);

    
     parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=EO(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
         final_curve(nrun,:)=iter(1:max_iter);
        z2(nrun) =  final;

    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
    curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA_2：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);
%     
 
    parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=SAO(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
         final_curve(nrun,:)=iter(1:max_iter);
        z2(nrun) =  final;

    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
    curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA_2：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);
%     
%    
% %     
%     
 
   
   
    parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=ACGRIME(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
        z2(nrun) =  final;
     final_curve(nrun,:)=iter(1:max_iter);
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);

  parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=IRIME(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
        z2(nrun) =  final;
     final_curve(nrun,:)=iter(1:max_iter);
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);

  parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=TERIME(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
        z2(nrun) =  final;
     final_curve(nrun,:)=iter(1:max_iter);
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);


  parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=EOSMA(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
        z2(nrun) =  final;
     final_curve(nrun,:)=iter(1:max_iter);
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);

  parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=RDGMVO(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
        z2(nrun) =  final;
     final_curve(nrun,:)=iter(1:max_iter);
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);

     parfor nrun=1:run
         run_seed=Rand_Seeds(1+mod(dim*func_num*run+nrun-run,length(Rand_Seeds)));
        rng(run_seed,'twister');
        [final,position,iter]=MTVSCA(pop_size,max_iter,lower_bound,upper_bound,dim,fobj);
        final_main(nrun)=final;
        z2(nrun) =  final;
     final_curve(nrun,:)=iter(1:max_iter);
    end
    box_plot = [box_plot;final_main]; %统计箱型图结果
        curve_plot=[curve_plot;mean(final_curve,1)];%统计迭代图结果
    zz = [min(final_main);std(final_main);mean(final_main);median(final_main);max(final_main)];
    resu = [resu,zz];
    rs = ranksum(z1,z2);
    if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rs=1;
    end
    rank_sum_resu = [rank_sum_resu,rs]; %统计秩和检验结果
        rk = kruskalwallis([z1' z2'],[],'off');
        if isnan(rs)  %当z1与z2完全一致时会出现NaN值，这种概率很小，但是要做一个防止报错
        rk=1;
    end
     kruskalwallis_sum_resu = [kruskalwallis_sum_resu,rk];   %统计秩和检验结果
    disp(['BMMPA：最优值:',num2str(zz(1)),' 标准差:',num2str(zz(2)),' 平均值:',num2str(zz(3)),' 中值:',num2str(zz(4)),' 最差值:',num2str(zz(5))]);

    

    rank_sum_RESULT = [rank_sum_RESULT;rank_sum_resu];  %统计秩和检验结果
      kruskalwallis_sum_RESULT=[kruskalwallis_sum_RESULT;kruskalwallis_sum_resu];  %统计秩和检验结果
              [rk,~,stasss] = kruskalwallis(box_plot',[],'off');
     kruskalwallis_Rank_RESULT=[kruskalwallis_Rank_RESULT;stasss.meanranks];
    RESULT = [RESULT;resu];   %统计标准差，平均值，最优值等结果
    RANK_mean=rank_mean_23(resu(3,:));%这个就是排名
    RESULT=[RESULT;RANK_mean];
    RANK_result=[RANK_result;RANK_mean];
          curve_plot_total(:,:,func_num)=curve_plot;
    total(:,:,func_num)=box_plot;
    
    
  
end

mean(RANK_result)


