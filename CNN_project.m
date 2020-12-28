close all force

project = fullfile('dataset','train');

imds = imageDatastore(project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

class(imds) % what kind of object is imds
%% 
 
imds;
labelCount = countEachLabel(imds);
unique(imds.Labels)
%%  
iimage=100;
img = imds.readimage(iimage); 
figure;
imshow(img,'initialmagnification',1000)
%%
figure
imshow(preview(imds),'initialmagnification',1000); %preview(imds) is the same as imds.readimage(1)
%%
% show the first 256 images 
im=imtile(imds.Files(1:256)); 
figure
imshow(im,'initialmagnification',200)
%%
% show the last 256 images
im=imtile(imds.Files(end-255:end));
figure
imshow(im,'initialmagnification',200)
%% 
% If needed, you can define a custom function applied to each image at time 
% of reading. For instance, to resize each image to 256x256:

% automatic resizing
imds.ReadFcn = @(x)imresize(imread(x),[64 64]);
%% 
% In the above code, the property |ReadFcn| is assigned a handle to an _inline 
% function_, whose argument is the filename |x|. 
% 
% To restore to default read function:

%imds.ReadFcn = @(x)imread(x);
%% 
% To perform an automatic rescaling of the values use the following syntax:

% automatic rescaling
divideby=255;
%imds.ReadFcn = @(x)double(imread(x))/divideby;

%% 
% In order to estimate the generalization capability during training, we need 
% to extract a valdation set from the provided training set. Let's take the 85% 
% of the images for actual training and the remaining 15% for validation.

% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%% Network design and training
% create the structure of the network
layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'WeightsInitializer','narrow-normal','Name','conv_1') 
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'WeightsInitializer','narrow-normal','Name','conv_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'WeightsInitializer','narrow-normal','Name','conv_3') 
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
    
%% 
    % training options
options = trainingOptions('sgdm', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 15,...
    'Verbose',false, ...
    'MiniBatchSize',32, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress')
%% 
% Then we train the network.

% train the net
net = trainNetwork(imdsTrain,layers,options);
% validation accuracy of about 25%

%% 
% However, let's evaluate the performance on the test set.

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

% point 2

%%
% data augmentation with reflection dx/sx
augmenter = imageDataAugmenter(...
    'RandXReflection', true)
auimdsTrain = augmentedImageDatastore([64 64 1],imdsTrain,'DataAugmentation',augmenter);

%%
% train the network with the new data 
% aunet = trainNetwork(auimdsTrain,layers,options);

% first attempt:
% +5% validation accuracy more or less
% it seems that better results would be obtained with more than 30 epochs

%%
% add batch normalization layers

layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'WeightsInitializer','narrow-normal','Name','conv_1')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')
    
    dropoutLayer(.1 , 'Name','dropout_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,16,'WeightsInitializer','narrow-normal','Name','conv_2')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    dropoutLayer(.1 , 'Name','dropout_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,32,'WeightsInitializer','narrow-normal','Name','conv_3')
    
    batchNormalizationLayer('Name', 'BN_3')
   
    reluLayer('Name','relu_3')
    
    dropoutLayer(.1 , 'Name','dropout_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    dropoutLayer(.2 , 'Name','dropout_4')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
   
%%
% new options

options = trainingOptions('adam', ...
'InitialLearnRate', 0.001, ...
'ValidationData',imdsValidation, ...
'MaxEpochs', 30, ...
'ValidationFrequency', 50, ...
'ValidationPatience', 5,...
'Verbose',false, ...
'MiniBatchSize',32, ...
'ExecutionEnvironment','parallel',...
'Plots','training-progress');
    
%%
% new training
    
aunet = trainNetwork(auimdsTrain,layers,options);

%%
% - changing the dimension of the filters and adding the batch normalization
% layers clearly improve the results (50% validation accuracy). A further improvement is provided by
% changing the InitialErrorRate parameter and replacing 'sgdm' with 'adam'
% (of about +5%)
% - with a single dropout layer of p = 0.5 after the fully connected layer
% there's no improvement
% - with dropout layers of p = 0.2 after the Relu layers and the dropout 
% layer after the fully connected one, the rusults are worse (40%)
% - with a dropout layer after the last Relu and a dropout after the fully
% connected the validation accuracy is of about 45%
% - In general the dropouts layers don't seem to provide big
% improvements. On the contrary, in many cases the results get worse (probably because of the batch
% normalization layers)
