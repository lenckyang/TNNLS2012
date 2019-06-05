
clear
for kkk=1:20
test=zeros(1,1);
train=zeros(1,1);
train_time=zeros(1,1);
testing_time=zeros(1,1);



for rnd = 1 : 50

    
load fried.mat;
data=fried';
data=mapminmax(data);
data(11,:)=(data(11,:)/2)+0.5;
data=[data(11,:); data(1:10,:)]';
rand_sequence=randperm(size(data,1));
    temp_data=data;
    data=temp_data(rand_sequence, :);
    Training=data(1:20768,:);
    Testing=data(20769:40768,:);

    
[train_time, test_time,  train_accuracy11, test_accuracy11]=B_ELM(Training,Testing,0,1,'sig',kkk);




    D_ELM_test(rnd,1)=test_accuracy11;
    D_ELM_train(rnd,1)=train_accuracy11;
    D_ELM_train_time(rnd,1)=train_time;
    

end


   DD_ELM_learn_time(kkk)=mean(D_ELM_train_time);
   

   
   DD_ELM_train_accuracy(kkk)=mean(D_ELM_train);
   
   DD_ELM_test_accuracy(kkk)=mean(D_ELM_test);
   
    

    
end



