%% Step 1: Define folders and load datastores
imageDir = 'Dataset/images';
maskDir  = 'Dataset/binary_masks';

imds = imageDatastore(imageDir);
classNames = ["background", "deck"];
labelIDs = [0, 255];
pxds = pixelLabelDatastore(maskDir, classNames, labelIDs);

%% Step 2: Split data: 80 train, 17 val, 16 test
numImages = numel(imds.Files);
rng(1); % reproducible split
indices = randperm(numImages);

idxTrain = indices(1:80);
idxVal   = indices(81:97);
idxTest  = indices(98:end);

imdsTrain = subset(imds, idxTrain);
pxdsTrain = subset(pxds, idxTrain);

imdsVal = subset(imds, idxVal);
pxdsVal = subset(pxds, idxVal);

imdsTest = subset(imds, idxTest);
pxdsTest = subset(pxds, idxTest);

%% Step 3: Define output size and folders
targetSize = [384 512];

outputBase = 'Dataset';
sets = {'train', 'val', 'test'};
imdsSets = {imdsTrain, imdsVal, imdsTest};
pxdsSets = {pxdsTrain, pxdsVal, pxdsTest};

for s = 1:3
    imageOutDir = fullfile(outputBase, sets{s}, 'images');
    maskOutDir  = fullfile(outputBase, sets{s}, 'masks');
    mkdir(imageOutDir); mkdir(maskOutDir);

    imdsS = imdsSets{s};
    pxdsS = pxdsSets{s};

    reset(imdsS);
    reset(pxdsS);

    for i = 1:numel(imdsS.Files)
        % Read image and mask
        I = readimage(imdsS, i);
        M = readimage(pxdsS, i);

        % Resize
        Iresized = imresize(I, targetSize);
        Mresized = imresize(M, targetSize, 'nearest');

        % Create filenames
        [~, baseName, ~] = fileparts(imdsS.Files{i});
        imwrite(Iresized, fullfile(imageOutDir, [baseName, '.jpg']));
        imwrite(uint8(Mresized == "deck") * 255, fullfile(maskOutDir, [baseName, '_mask.png']));
    end

    fprintf('Saved %s set: %d images\n', sets{s}, numel(imdsS.Files));
end

%% Step 4: Create image and label datastores from resized data
root = 'Dataset';
sets = {'train', 'val', 'test'};

for s = 1:3
    setName = sets{s};

    imageDir = fullfile(root, setName, 'images');
    maskDir  = fullfile(root, setName, 'masks');

    imds = imageDatastore(imageDir);

    classNames = ["background", "deck"];
    labelIDs   = [0, 255];

    pxds = pixelLabelDatastore(maskDir, classNames, labelIDs);

    ds = pixelLabelImageDatastore(imds, pxds);

    switch setName
        case 'train'
            trainDS = ds;
        case 'val'
            valDS = ds;
        case 'test'
            testDS = ds;
    end
end
fprintf('Step 4 done!');
%% Step 5: Define DeepLab v3+ Network
inputSize = [384 512 3];  % Matches your resized images
numClasses = 2;

lgraph = deeplabv3plusLayers(inputSize, numClasses, 'resnet18');
fprintf('Step 5 done!');
%% Step 6: Define Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 10, ...
    'ValidationData', valDS, ...
    'Plots', 'training-progress');
fprintf('Step 6 done!');
%% Step 7: Define Data Augmentation (Train Only)
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

% Recreate imds and pxds from saved folders
imdsTrain = imageDatastore(fullfile(root, 'train', 'images'));
pxdsTrain = pixelLabelDatastore(fullfile(root, 'train', 'masks'), classNames, labelIDs);

augmentedTrainDS = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'DataAugmentation', augmenter);
fprintf('Step 7 done!');
%% Step 8: Train the Network
net = trainNetwork(augmentedTrainDS, lgraph, options);
fprintf('Model trained successfully');