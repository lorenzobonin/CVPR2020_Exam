close all force

project = fullfile('dataset','train');

imds = imageDatastore(project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

class(imds) % what kind of object is imds
%% 
 
imds
labelCount = countEachLabel(imds)
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
% Some random instances of the training set, with corresponding label:

% show some instances
figure;
perm = randperm(length(imds.Labels),20);
for ii = 1:20
    subplot(4,5,ii);
    imshow(imds.Files{perm(ii)}); 
    title(imds.Labels(perm(ii)));
end
sgtitle('some instances of the training set')
%% 
% In order to estimate the generalization capability during training, we need 
% to extract a valdation set from the provided training set. Let's take the 85% 
% of the images for actual training and the remaining 15% for validation.

% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize')

%% Network design and training
% create the structure of the network
layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1') 
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3') 
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
    
    layers(1).Mean = 0;
    layers(1).StandardDeviation = 0.01;
    layer(2).Bias = 0;
%% 
    % training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf,...
    'Verbose',false, ...
    'MiniBatchSize',32, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress')


%% 
% Then we train the network.

% train the net
net = trainNetwork(imdsTrain,layers,options);

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