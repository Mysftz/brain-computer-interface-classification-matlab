function [sess_accuracy, predicted_object] = BCI_CNN_Function(event, label, runs, sess, sub, target, testData, trainData)

%PRE-PROCESSING:
% Pre-stimulus mean removal:
mn=mean(trainData(:,1:100,:),2);
trainDatamn=trainData-mn;

% Normalisation:
for j=1:1600
    for i=1:8
        datemp=trainDatamn(i,101:350,j);
        data(i,:,1,j)=normalize(datemp,'range');
    end
end

labeldata=categorical(target); %change to categorical

% CONOVOLUTIONAL NEURAL NETWORK (CNN) LAYERS:
layers = [ ...
    imageInputLayer([8 250 1])

    convolution2dLayer(3,128,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    %dropoutlayer was originally here
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.1)

    convolution2dLayer(3,64,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    %dropoutlayer was originally here
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.1)
    
    convolution2dLayer(3,32,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    %dropoutlayer was originally here
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.1)
    
    fullyConnectedLayer(64)
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ... %Solver
    'InitialLearnRate',0.005, ...
    'MaxEpochs',30, ... %How many passes of the training set is complete, lower value = faster run time
    'Shuffle','every-epoch', ... %No default but set so the data is shuffled before each epoch
    'Verbose',false, ... %Set to 0 (false) as default is true
    'Plots','none'); %Set to none as default is training-progress
    %'Momentum', 0.9000, ...  % ONLY USED WITH SGDM SOLVER
    %'LearnRateSchedule', 'none', ... %Alters the IntialLearnRate while training
    %'MiniBatchSize', 128, ... %Used to evaluate loss (Future recomendation)
    %'gradientdecayfactor', 0.9% ...  % ONLY USED WITH ADAM SOLVER

% TRAINING THE CNN:
[net,info] = trainNetwork(data, labeldata, layers, options);

flashes=8*runs;
blocks=50;
mn=mean(testData(:,1:100,:),2);
testDatamn=testData-mn;

%Normalise  the data again which is dependent on the runs_per_block of each seesion.
for j=1:blocks*flashes
    for i=1:8
        datempt=testDatamn(i,101:350,j);
        testdata(i,:,1,j)=normalize(datempt,'range');
    end
end

%Average probability sorting
[predicted, prob]= classify(net, testdata);

predicted_testevent=[event, prob]; 

%sorted method, averaged prob output
for i=1:blocks
sorted=sortrows(predicted_testevent((i-1)*flashes+1:i*flashes,:),1); %Sort according to 1st column
    for j=1:flashes/runs
    test_prediction(j,:)=median(sorted((j-1)*runs+1:j*runs,:));
    end
    [val, ind]=max(test_prediction);
    predicted_object(i)=ind(3); %Find max target
end

success=0;
for i=1:blocks
    if (predicted_object(i)==label(i))
        success(i)=1;
    end
end

sess_accuracy=(sum(success)/blocks)*100; %Calculates session accuracy