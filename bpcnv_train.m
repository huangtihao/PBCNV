%% �ô���Ϊ����BP�������Ԥ���㷨
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>

%% ��ջ�������
clc
clear
%% ѵ������Ԥ��������ȡ����һ��
%���������������

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
%����groudtruth��������1��ʾgain��2��ʾhemi_loss��3��ʾhomo_loss��
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

%�ڵ����
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

%��1��trainlines���������
k=rand(1,trainLines);
[m,n]=sort(k);
%�õ������������
ginput=gdata(:,column);
goutput1 =gdata(:,6);
%�����һά�����ά��0������1gain��2hemi_loss��3homo_loss;
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
%�ҳ�ѵ�����ݺ�Ԥ������

ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';


%��������������ݹ�һ��
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);




%% BP����ѵ��
% %��ʼ������ṹ
net=newff(ginputn,goutput_train,hiddennum);

%�ڵ�����
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;
c2 = 1.49445;

maxgen=1;   % ��������  
sizepop=30;   %��Ⱥ��ģ

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

for i=1:sizepop
    pop(i,:)=5*rands(1,numsum);
    V(i,:)=rands(1,numsum);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
end


% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest=pop;    %�������
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for i=1:maxgen
    i;
    
    for j=1:sizepop
        
        %�ٶȸ���
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %��Ⱥ����
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %����Ӧ����
        pos=unidrnd(49);
        if rand>0.90
            pop(j,pos)=5*rands(1,1);
        end
      
        %��Ӧ��ֵ
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    end
    
    for j=1:sizepop
    %�������Ÿ���
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %Ⱥ�����Ÿ��� 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

% %% �������
% plot(yy)
% title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
% xlabel('��������');ylabel('��Ӧ��');

x=zbest;
%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
B2=B2';

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
net.trainParam.epochs=2000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

%����ѵ��
[net,per2]=train(net,ginputn,goutput_train);
save ('PSOBP','net');

% %% BP����Ԥ��
% %���ݹ�һ��
% inputn_test=mapminmax('apply',input_test,inputps);
% an=sim(net,inputn_test);
% test_simu=mapminmax('reverse',an,outputps);
% error=test_simu-output_test;












% net.trainParam.epochs=200;
% net.trainParam.lr=0.1;
% net.trainParam.goal=1e-4;
% 
% %����ѵ��
% net=train(net,ginputn,goutput_train);
% 
% save ('PSOBP','net');
% %======================================

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>
