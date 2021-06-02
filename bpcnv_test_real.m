%%加载神经网络
clc
clear


data_truth1=load('data\NA19238.mat');
data_truth=data_truth1.('NA19238');

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
data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6];
column=[2,3,4,5];
[m1,n1] = size(data_trains);
[m3,n3]=size(data_truth);
truthLines=m3;
trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);
gdatatruth(1:truthLines,:)=data_truth(1:truthLines,:);
column3=[1,2];
gdatatruth_bin=gdatatruth(:,column3);
gdatatruth_bin=gdatatruth_bin(:);
gdatatruth_bin=gdatatruth_bin';
%从1到trainLines间随机排序
k=rand(1,trainLines);
[m,n]=sort(k);
%得到输入输出数据
ginput=gdata(:,column);
goutput1 =gdata(:,6);
%输出从一维变成四维
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


%选连样本输入输出数据归一化
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%%加载网络
load('-mat','PSOBP');

column=[2,3,4,5];
num=1;
TP_count_sum=0;
TPFP_count_sum=0;

datat=load('data\NA19238_trains.mat');
data_tests = datat.('NA19238_trains');
[m2,n2] = size(data_tests);
testLines=m2;
gdata2(1:testLines,:) = data_tests(1:m2,:);
ginput2_bin=gdata2(:,1);
ginput2=gdata2(:,column);
goutput1 =gdata2(:,6);
goutput2=zeros(testLines,4);
for i=1:testLines
    switch goutput1(i)
        case 0
            goutput2(i,:)=[1 0 0 0];
        case 1
            goutput2(i,:)=[0 1 0 0];
        case 2
            goutput2(i,:)=[0 0 1 0];
    case 3
            goutput2(i,:)=[0 0 0 1];
    end
end
ginput_test=ginput2((1:testLines),:)';
goutput_test=goutput2((1:testLines),:)';
%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',ginput_test,ginputps);

%网络预测输出
an=sim(net,inputn_test);

%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

%% 结果分析

% figure(1)
% plot(BPoutput,':og')
% hold on
% plot(goutput_test,'-*');
% legend('预测输出','期望输出')
% title('BP网络预测输出','fontsize',12)
% ylabel('函数输出','fontsize',12)
% xlabel('样本','fontsize',12)

%预测误差
error=BPoutput-goutput_test;

% figure(2)
% plot(error,'-*')
% title('BP网络预测误差','fontsize',12)
% ylabel('误差','fontsize',12)
% xlabel('样本','fontsize',12)
% 
% figure(3)
% plot((goutput_test-BPoutput)./BPoutput,'-*');
% title('神经网络预测误差百分比')
abs_error=abs(error);
errorsum=sum(abs(error));

    %% 结果分析

for i=1:m2
    abs_error1(i)=find(abs_error(:,i)==min(abs_error(:,i)));
end

for i=1:m2
    output_test1(i)=find(goutput_test(:,i)==max(goutput_test(:,i)));
end


% figure(1)
% plot(abs_error1(:,1:m2),':og')
% hold on
% plot(output_test1(:,1:m2),'-*');
% legend('预测输出','期望输出')
% title('BP网络预测输出','fontsize',12)
% ylabel('函数输出','fontsize',12)
% xlabel('样本','fontsize',12)

disp('-------------');


fid=fopen('NA19238bin.txt','wt'); %将bin_number数组写入文件
TP_count=0;
P_count=0;
TPFP_count=0;
k=1;
for q=1:testLines
    if ( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));
        fprintf(fid,'gain\t1');
        fprintf(fid,'\n');
        binnumber(k)=ginput2_bin(q);
        k=k+1;
    end
    if ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));%将bin_number写入文件
        fprintf(fid,'hemi_loss\t2');
        fprintf(fid,'\n');
        binnumber(k)=ginput2_bin(q);
        k=k+1;
    end
    if ( abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));%将bin_number写入文件
        fprintf(fid,'homo_loss\t3');
        fprintf(fid,'\n');
        binnumber(k)=ginput2_bin(q);
        k=k+1;

    end
    if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)) 
        TP_count=TP_count+1;
    end
    if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
        P_count=P_count+1;
    end
    if ( (abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) )
        TPFP_count=TPFP_count+1;
    end
end
fclose(fid);
bound=0;
[m4,n4]=size(binnumber);
for u=1:n4
    if(ismember(binnumber(u),gdatatruth_bin)==1)
        bound=bound+1;
    end
end
TPFP_count_sum=TPFP_count_sum+TPFP_count;
TP_count_sum=TP_count_sum+TP_count;

TP_count_avg=TP_count_sum/num;
TPFP_count_avg=TPFP_count_sum/num;
recall=TP_count_avg/P_count;
precision=TP_count_avg/TPFP_count_avg;
F1_score=(2*recall*precision)/(recall+precision);
disp('sensitivity:');
disp(recall);
disp('precision:');
disp(precision);
disp('F1-score:');
disp(F1_score);
