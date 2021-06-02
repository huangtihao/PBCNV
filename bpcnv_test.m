%%����������
clc
clear

data_truth1=load('groundtruth.mat');
data_truth=data_truth1.('groundtruth');

data1=load('data\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('data\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data3=load('data\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('data\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('data\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data6=load('data\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1 = data1.('sim1_4_4100_read_trains');
data_trains2 = data2.('sim1_6_6100_read_trains');
data_trains3 = data3.('sim1_4_4100_read_trains');
data_trains4 = data4.('sim1_6_6100_read_trains');
data_trains5 = data5.('sim1_4_4100_read_trains');
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
% gdatatruth_bin=gdatatruth(:,column3);
groundtruth=data_truth(:,column3); 
gtRev=fliplr(groundtruth(:,2)'); 
% gdatatruth_bin=gdatatruth_bin(:);
% gdatatruth_bin=gdatatruth_bin';
%��1��trainlines���������
k=rand(1,trainLines);
[m,n]=sort(k);
%�õ������������
ginput=gdata(:,column);
goutput1 =gdata(:,6);
%�����һά�����ά
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

%%��������
load('-mat','PSOBP');

column=[2,3,4,5];
num=50;
TP_count_sum=0;
TPFP_count_sum=0;

bound = 0;
count_bias_sum=0;
boundary=[]; %ÿ��bam�ı߽磬����������ͼ
num_boundary=0;
TPR=[];
FPR=[];


for t=1:num
    data2=load(['data\0.2_6x_mat\sim', num2str(t) ,'_6_6100_read_trains.mat']);
    data_tests = data2.(['sim', num2str(t) ,'_6_6100_read_trains']);
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
    %% BP����Ԥ��
    %Ԥ�����ݹ�һ��
    inputn_test=mapminmax('apply',ginput_test,ginputps);

    %����Ԥ�����
    an=sim(net,inputn_test);

    %�����������һ��
    BPoutput=mapminmax('reverse',an,outputps);

    %Ԥ�����
    error=BPoutput-goutput_test;
    abs_error=abs(error);
    errorsum=sum(abs(error));


    fid=fopen(['0.2_6x_binnumber\sim', num2str(t) ,'_6_6100_bin_number.txt'],'wt'); %��bin_number����д���ļ�
    TP_count=0;
    P_count=0;
    TPFP_count=0;
    k=1;
    kkk=1;
    binnumber=[];
    binnumber_tpfp=[];
    for q=1:testLines
%         if ( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1)
%             fprintf(fid,'%d\t',ginput2_bin(q));
%             fprintf(fid,'gain\t1');
%             fprintf(fid,'\n');
%             binnumber(k)=ginput2_bin(q);
%             k=k+1;
%         end
%         if ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1)
%             fprintf(fid,'%d\t',ginput2_bin(q));%��bin_numberд���ļ�
%             fprintf(fid,'hemi_loss\t2');
%             fprintf(fid,'\n');
%             binnumber(k)=ginput2_bin(q);
%             k=k+1;
%         end
%         if ( abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)
%             fprintf(fid,'%d\t',ginput2_bin(q));%��bin_numberд���ļ�
%             fprintf(fid,'homo_loss\t3');
%             fprintf(fid,'\n');
%             binnumber(k)=ginput2_bin(q);
%             k=k+1;
%         end
        

        if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)) 
            TP_count=TP_count+1;
            binnumber(k)=ginput2_bin(q);
            k=k+1;
        end
        if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
            P_count=P_count+1;
            binnumber_tpfp(kkk)=ginput2_bin(q);
            kkk=kkk+1;
        end
        if (  (abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q)) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q)) )
            TPFP_count=TPFP_count+1;
        end
    end
    fclose(fid);
    
    % TPR�ļ���
    TPR(t)=TP_count./TPFP_count;
    % FPR�ļ���
    FPR(t)=(TPFP_count-TP_count)./(1512*14);

    TPFP_count_sum=TPFP_count_sum+TPFP_count;
    TP_count_sum=TP_count_sum+TP_count;
    boundbias = (P_count - TP_count)./14;
    bound = bound + boundbias;
    
        % ����⵽��14��cnv��bin��ͷ��β�ı�ű�������
    binnumberRev=fliplr(binnumber);
    [m3,n3]=size(binnumber);
    jj=1;
    bin1=[];
    for ii=1:n3
        if binnumber(ii) >= groundtruth(jj,1)
            bin1(jj)=binnumber(ii);
            jj=jj+1;
        end
        if jj>14
           break; 
        end
    end
    jjj=1;
    bin22=[];
    for iii=1:n3
        if binnumberRev(iii) <= gtRev(jjj)
            bin22(jjj)=binnumberRev(iii);
            jjj=jjj+1;
        end
        if jjj>14
           break; 
        end
    end
    bin2=fliplr(bin22);
    [m5,n5]=size(bin1);
    [m6,n6]=size(bin2);
    if(n5<14 || n6<14)
       continue; 
    end
    bin=[bin1;bin2]'; 
    % ����߽羫��
    [m4,n4]=size(bin);
    c_bias1=[];
    c_bias2=[];
    for rr=1:m4
        c_bias1(rr)=bin(rr,1)-groundtruth(rr,1);
        c_bias2(rr)=groundtruth(rr,2)-bin(rr,2);
    end
    c_bias=sum(c_bias1)+sum(c_bias2);
    count_bias=c_bias./14;
    if count_bias > 50
        continue;
    end
    boundary(t)=count_bias;
    num_boundary = num_boundary + 1;
    count_bias_sum = count_bias_sum+count_bias;
    binnumber_fp=setdiff(binnumber_tpfp,binnumber);

    
    
    
    
    
    
    
%     bound=0;
%     [m4,n4]=size(binnumber);
%     for u=1:n4
%         if(ismember(binnumber(u),gdatatruth_bin)==1)
%             bound=bound+1;
%         end
%     end
%     TPFP_count_sum=TPFP_count_sum+TPFP_count;
%     TP_count_sum=TP_count_sum+TP_count;
%     
    
    
        %% �������
        
    for i=1:m2
        abs_error1(i)=find(abs_error(:,i)==min(abs_error(:,i)));
    end

    for i=1:m2
        output_test1(i)=find(goutput_test(:,i)==max(goutput_test(:,i)));
    end


%     figure(1)
%     plot(abs_error1(:,15000:m2),':og')
%     hold on
%     plot(output_test1(:,15000:m2),'-*');
%     legend('Ԥ�����','�������')
%     title('BP����Ԥ�����','fontsize',12)
%     ylabel('�������','fontsize',12)
%     xlabel('����','fontsize',12)

%    disp('-------------');
% 
%     figure(2)
%     plot(error,'-*')
%     title('BP����Ԥ�����','fontsize',12)
%     ylabel('���','fontsize',12)
%     xlabel('����','fontsize',12)
%     
%     figure(3)
%     plot((goutput_test-BPoutput)./BPoutput,'-*');
%     title('������Ԥ�����ٷֱ�')
    
end

TP_count_avg=TP_count_sum/num;
TPFP_count_avg=TPFP_count_sum/num;
recall=TP_count_avg/P_count;
precision=TP_count_avg/TPFP_count_avg;
F1_score=(2*recall*precision)/(recall+precision);
boundary_bias=bound/num;
count_bias_avg=count_bias_sum/num_boundary;
disp('sensitivity:');
disp(recall);
disp('precision:');
disp(precision);
disp('F1-score:');
disp(F1_score);
disp('all_bound_bias:');
disp(boundary_bias);
disp('14_bound_bias:');
disp(count_bias_avg);
disp('bound_bias:');
disp(boundary);
disp('TPR:');
disp(TPR);
disp('FPR:');
disp(FPR);
