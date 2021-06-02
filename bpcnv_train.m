%% 该代码为基于BP神经网络的预测算法
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">该案例作者申明：</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1：本人长期驻扎在此<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">板块</font></a>里，对该案例提问，做到有问必答。本套书籍官方网站为：<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2：点此<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">从当当预定本书</a>：<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">《Matlab神经网络30个案例分析》</a>。</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">：此案例有配套的教学视频，视频下载方式<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">。 </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4：此案例为原创案例，转载请注明出处（《Matlab神经网络30个案例分析》）。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5：若此案例碰巧与您的研究有关联，我们欢迎您提意见，要求等，我们考虑后可以加在案例里。</font></span></td>	</tr>		</table>
% </html>

%% 清空环境变量
clc
clear
%% 训练数据预测数据提取及归一化
%下载输入输出数据

% 
% for k =1:50
%     load(['F:\matlab2016a\bin\MFCNV_BPNN\0.2_4x\sim',num2str(k),'_4_4100_read_trains.txt']);
%     save(['F:\matlab2016a\bin\MFCNV_BPNN\0.2_4x_mat\sim',num2str(k),'_4_4100_read_trains.mat'],['sim',num2str(k),'_4_4100_read_trains']);
%     load(['F:\matlab2016a\bin\MFCNV_BPNN\0.2_6x\sim',num2str(k),'_6_6100_read_trains.txt']);
%     save(['F:\matlab2016a\bin\MFCNV_BPNN\0.2_6x_mat\sim',num2str(k),'_6_6100_read_trains.mat'],['sim',num2str(k),'_6_6100_read_trains']);
%     load(['F:\matlab2016a\bin\MFCNV_BPNN\0.3_4x\sim',num2str(k),'_4_4100_read_trains.txt']);
%     save(['F:\matlab2016a\bin\MFCNV_BPNN\0.3_4x_mat\sim',num2str(k),'_4_4100_read_trains.mat'],['sim',num2str(k),'_4_4100_read_trains']);
%     load(['F:\matlab2016a\bin\MFCNV_BPNN\0.3_6x\sim',num2str(k),'_6_6100_read_trains.txt']);
%     save(['F:\matlab2016a\bin\MFCNV_BPNN\0.3_6x_mat\sim',num2str(k),'_6_6100_read_trains.mat'],['sim',num2str(k),'_6_6100_read_trains']);
%     load(['F:\matlab2016a\bin\MFCNV_BPNN\0.4_4x\sim',num2str(k),'_4_4100_read_trains.txt']);
%     save(['F:\matlab2016a\bin\MFCNV_BPNN\0.4_4x_mat\sim',num2str(k),'_4_4100_read_trains.mat'],['sim',num2str(k),'_4_4100_read_trains']);
%     load(['F:\matlab2016a\bin\MFCNV_BPNN\0.4_6x\sim',num2str(k),'_6_6100_read_trains.txt']);
%     save(['F:\matlab2016a\bin\MFCNV_BPNN\0.4_6x_mat\sim',num2str(k),'_6_6100_read_trains.mat'],['sim',num2str(k),'_6_6100_read_trains']);
% end
%加载groudtruth，第四列1表示gain、2表示hemi_loss、3表示homo_loss。
% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\groundtruth.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\groundtruth.mat','groundtruth');

% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19238.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19238.mat','NA19238');
% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19239.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19239.mat','NA19239');
% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19240.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19240.mat','NA19240');
% 
% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19238_tests.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19238_tests.mat','NA19238_tests');
% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19239_tests.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19239_tests.mat','NA19239_tests');
% load('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19240_tests.txt');
% save('D:\Matlab\bin\matlab_procedure\MFCNV_BPNN\NA19240_tests.mat','NA19240_tests');

%节点个数
inputnum=4;
hiddennum=10;
outputnum=4;


data1=load('data\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('data\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data3=load('data\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('data\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('data\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data6=load('data\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1 = data1.('sim1_4_4100_read_trains');
data_trains2 = data2.('sim1_4_4100_read_trains');
data_trains3 = data3.('sim1_4_4100_read_trains');
data_trains4 = data4.('sim1_6_6100_read_trains');
data_trains5 = data5.('sim1_6_6100_read_trains');
data_trains6 = data6.('sim1_6_6100_read_trains');
data_trains=[data_trains1;data_trains2;data_trains3;data_trains5;data_trains6;];
column=[2,3,4,5];
[m1,n1] = size(data_trains);

trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);

%从1到trainlines间随机排序
k=rand(1,trainLines);
[m,n]=sort(k);
%得到输入输出数据
ginput=gdata(:,column);
goutput1 =gdata(:,6);
%输出从一维变成四维：0正常，1gain，2hemi_loss，3homo_loss;
goutput=zeros(trainLines,4);
for i=1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:)=[1 0 0 0];
        case 1
            goutput(i,:)=[0 1 0 0];
        case 2
            goutput(i,:)=[0 0 1 0];
        case 3
            goutput(i,:)=[0 0 0 1];
    end
end
%找出训练数据和预测数据

ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';


%样本输入输出数据归一化
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);




%% BP网络训练
% %初始化网络结构
net=newff(ginputn,goutput_train,hiddennum);

%节点总数
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

% 参数初始化
%粒子群算法中的两个参数
c1 = 1.49445;
c2 = 1.49445;

maxgen=1;   % 进化次数  
sizepop=30;   %种群规模

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

for i=1:sizepop
    pop(i,:)=5*rands(1,numsum);
    V(i,:)=rands(1,numsum);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
end


% 个体极值和群体极值
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest=pop;    %个体最佳
fitnessgbest=fitness;   %个体最佳适应度值
fitnesszbest=bestfitness;   %全局最佳适应度值

%% 迭代寻优
for i=1:maxgen
    i;
    
    for j=1:sizepop
        
        %速度更新
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %种群更新
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %自适应变异
        pos=unidrnd(49);
        if rand>0.90
            pop(j,pos)=5*rands(1,1);
        end
      
        %适应度值
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    end
    
    for j=1:sizepop
    %个体最优更新
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %群体最优更新 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

% %% 结果分析
% plot(yy)
% title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
% xlabel('进化代数');ylabel('适应度');

x=zbest;
%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
B2=B2';

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP网络训练
%网络进化参数
net.trainParam.epochs=2000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

%网络训练
[net,per2]=train(net,ginputn,goutput_train);
save ('PSOBP','net');

% %% BP网络预测
% %数据归一化
% inputn_test=mapminmax('apply',input_test,inputps);
% an=sim(net,inputn_test);
% test_simu=mapminmax('reverse',an,outputps);
% error=test_simu-output_test;












% net.trainParam.epochs=200;
% net.trainParam.lr=0.1;
% net.trainParam.goal=1e-4;
% 
% %网络训练
% net=train(net,ginputn,goutput_train);
% 
% save ('PSOBP','net');
% %======================================

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab神经网络30个案例分析</a></font></p><p align="left"><font size="2">相关论坛：</font></p><p align="left"><font size="2">《Matlab神经网络30个案例分析》官方网站：<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab技术论坛：<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab函数百科：<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab中文论坛：<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>
