

% ----------------------- README ------------------------------------------
%     最后一次修改     ：2024/7/20
%                         欢迎关注₍^.^₎♡
%                    ------------
%     项目             ：JADE算法
%                    ------------
%     参考文献         ：J. Zhang and A. C. Sanderson, "JADE: adaptive differential evolution
%                       with optional external archive," IEEE Trans. Evolut. Comput., vol. 13,
%                       no. 5, pp. 945-958, 2009.
%     微信公众号/CSDN/知乎/B站 ：KAU的云实验台
%     定制或答疑加微信 : KAUsysyt
%     付费代码(更全)   ：https://mbd.pub/o/author-a2iWlGtpZA==
%     免费代码         ：公众号后台回复"资源"
% -------------------------------------------------------------------------

function [fitnessBestP,bestP,Convergence_curve]=JADE(SearchAgents_no,Gmax,lb,ub,dim,fobj)


fitnessBestP=inf;
G=0;%设置迭代器（当前迭代代数）
lu = [lb .* ones(1, dim); ub .* ones(1, dim)];
lb = lb .* ones(1, dim);
ub = ub .* ones(1, dim);
c = 1/10;% 控制因子
p = 0.05;
top=round(p*SearchAgents_no);%每代中最优的top个
A=[];%初始归档种群为空集
t=1;%记录归档种群A中的个体个数

% 初始 CR F
uCR=0.5;%初始化交叉概率
uF=0.5;%初始化缩放因子

% 种群初始化
P = repmat(lu(1, :), SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* (repmat(lu(2, :) - lu(1, :), SearchAgents_no, 1));
fitnessP=zeros(1,SearchAgents_no);
for i=1:SearchAgents_no
    fitnessP(i)=fobj(P(i,:));
end

% 主迭代
while G < Gmax

    Scr=[];%初始成功参加变异的个体的交叉概率为空集
    Sf=[];%初始成功参加变异的个体的缩放因子为空集
    n1=1;%记录Scr中的元素个数
    n2=1;%记录Sf中的元素个数

    % 更新适应度和CR F
    for i=1:SearchAgents_no
        fitnessP(i)=fobj(P(i,:));
        % 更新
        CR(i)=0.5;
        F(i)=0.5;
        while (CR(i)>1||CR(i)<0)
            CR(i)=normrnd(uCR,0.1);
        end
        if (F(i)>1)
            F(i)=1;
        end
        while (F(i)<=0)
            F(i)=cauchyrnd(uF,0.1);
        end
        if fitnessP(i)<fitnessBestP
            fitnessBestP=fitnessP(i);
        bestP=P(i,:);
        end
        
        
                    G=G+1;
    Convergence_curve(G) = fitnessBestP;
                    if G>Gmax
              break;
          end
        
        
    end

    % 获得种群前p个个体，方便后面随机选择
    [fitnessBestP,indexBestP]=min(fitnessP);
    bestP=P(indexBestP,:);
    [~,indexSortP]=sort(fitnessP);
    for i=1:top
        bestTopP(i,:)=P(indexSortP(i),:);
    end

    % 变异操作
    for i=1:SearchAgents_no
        %从top个个体中随机选出一个作为Xpbest
        k0=randperm(top,1);
        Xpbest=bestTopP(k0,:);

        %从当前种群P中随机选出P1
        k1=randi(SearchAgents_no);
        P1=P(k1,:);
        while (k1==i||k1==k0)
            k1=randi(SearchAgents_no);
            P1=P(k1,:);
        end

        %从P∪A中随机选出P2
        PandA=[P;A];
        [num,~]=size(PandA);
        k2=randi(num);
        P2=PandA(k2,:);
        while (k2==i||k2==k0||k2==k1)
            k2=randi(num);
            P2=PandA(k2,:);
        end

        % DE/current-to-pbest/1
        V(i,:)=P(i,:)+F(i).*(Xpbest-P(i,:))+F(i).*(P1-P2);
    end

    % 交叉操作
    for i=1:SearchAgents_no
        jrand=randi([1,dim]);
        for j=1:dim
            k3=rand;
            if(k3<=CR(i)||j==jrand)
                U(i,j)=V(i,j);
            else
                U(i,j)=P(i,j);
            end
        end
    end

    % 边界处理
    for i=1:SearchAgents_no
        for j=1:dim
            while (U(i,j)>ub(j) || U(i,j)<lb(j))
                U(i,j)=(ub(j)-lb(j))*rand+lb(j);
            end
        end
    end

    % 选择操作
    for i=1:SearchAgents_no
        fitnessU(i)=fobj(U(i,:));
        if(fitnessU(i)<fitnessP(i))
            A(t,:)=P(i,:);% 淘汰的个体放在A
            P(i,:)=U(i,:);% 更新
            fitnessP(i)=fitnessU(i);
            Scr(n1)=CR(i);
            Sf(n2)=F(i);
            t=t+1;
            n1=n1+1;
            n2=n2+1;
            if(fitnessU(i)<fitnessBestP)
                fitnessBestP=fitnessU(i);
                bestP=U(i,:);
            end
        end
        
            G=G+1;
    Convergence_curve(G) = fitnessBestP;
                    if G>Gmax
              break;
          end
        
    end

    %判断归档种群A的规模是否在NP之内，若大于，则随机移除个体使其规模保持NP
    [tA,~]=size(A);
    if tA>SearchAgents_no
        nRem=tA-SearchAgents_no;
        k4=randperm(tA,nRem);
        A(k4,:)=[];
        [tA,~]=size(A);
        t=tA+1;
    end

    %自适应参数，更新uCR和uF
    [~,ab]=size(Scr);
    if ab~=0
        newSf=(sum(Sf.^2))/(sum(Sf));
        uCR=(1-c)*uCR+c.*mean(Scr);
        uF=(1-c)*uF+c.*newSf;
    end




end

end

