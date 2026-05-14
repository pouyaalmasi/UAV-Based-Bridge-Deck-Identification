%% inference_deck.m
% Bridge Deck Identification using Trained DeepLab v3+
%
% Author: Pouya Almasi
%
% Description:
% This script loads a trained DeepLab v3+ model for bridge deck
% identification, performs semantic segmentation on either a single UAV
% image or a test dataset, generates predicted deck masks and overlays,
% and saves the results.

clc;
clear;
close all;

%% User Inputs

modelPath = "saved_models/deeplabv3plus_bridge_deck.mat";

% Single-image inference
imagePath = "sample_images/sample1.jpg";
outputDir = "sample_results";

% Batch test-set inference
testImageDir = "Dataset/test/images";
testMaskDir  = "Dataset/test/masks";

runSingleImageInference = true;
runBatchTestInference   = false;

%% Create Output Directory

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

%% Load Trained Model

data = load(modelPath);

if isfield(data, "net")
    net = data.net;
else
    error("The MAT file must contain a trained network variable named 'net'.");
end

%% Class Information

classNames = ["background", "deck"];
labelIDs   = [0 255];

%% Single-Image Inference

if runSingleImageInference

    fprintf("\nRunning single-image bridge deck segmentation...\n");

    I = imread(imagePath);

    predictedLabels = semanticseg(I, net);

    deckMask = predictedLabels == "deck";

    overlay = labeloverlay( ...
        I, ...
        predictedLabels, ...
        "IncludedLabels", "deck", ...
        "Colormap", [0 1 0], ...
        "Transparency", 0.45);

    %% Display Results

    figure("Name", "Bridge Deck Identification Results");

    subplot(1,3,1);
    imshow(I);
    title("Original Image");

    subplot(1,3,2);
    imshow(deckMask);
    title("Predicted Deck Mask");

    subplot(1,3,3);
    imshow(overlay);
    title("Predicted Deck Overlay");

    %% Save Outputs

    [~, imageName, ~] = fileparts(imagePath);

    maskPath = fullfile(outputDir, imageName + "_predicted_deck_mask.png");
    overlayPath = fullfile(outputDir, imageName + "_deck_overlay.png");

    imwrite(uint8(deckMask) * 255, maskPath);
    imwrite(overlay, overlayPath);

    fprintf("Single-image inference complete.\n");
    fprintf("Predicted mask saved to: %s\n", maskPath);
    fprintf("Overlay saved to: %s\n", overlayPath);

end

%% Batch Test-Set Inference and Visualization

if runBatchTestInference

    fprintf("\nRunning batch inference on test dataset...\n");

    testImds = imageDatastore(testImageDir);

    testPxds = pixelLabelDatastore( ...
        testMaskDir, ...
        classNames, ...
        labelIDs);

    testDS = pixelLabelImageDatastore(testImds, testPxds);

    predictedLabels = semanticseg( ...
        testDS, ...
        net, ...
        "MiniBatchSize", 4, ...
        "Verbose", false);

    numImagesToDisplay = min(16, numel(testImds.Files));

    for i = 1:numImagesToDisplay

        I  = readimage(testImds, i);
        GT = readimage(testPxds, i);
        P  = readimage(predictedLabels, i);

        predictedDeckMask = P == "deck";
        groundTruthDeckMask = GT == "deck";

        overlay = labeloverlay( ...
            I, ...
            P, ...
            "IncludedLabels", "deck", ...
            "Colormap", [0 1 0], ...
            "Transparency", 0.45);

        figure("Name", "Test Dataset Prediction");

        subplot(1,3,1);
        imshow(I);
        title("Original Image");

        subplot(1,3,2);
        imshow(uint8(groundTruthDeckMask) * 255);
        title("Ground Truth Mask");

        subplot(1,3,3);
        imshow(overlay);
        title("Predicted Overlay");

        [~, imageName, ~] = fileparts(testImds.Files{i});

        imwrite( ...
            uint8(predictedDeckMask) * 255, ...
            fullfile(outputDir, imageName + "_predicted_deck_mask.png"));

        imwrite( ...
            overlay, ...
            fullfile(outputDir, imageName + "_deck_overlay.png"));

    end

    fprintf("Batch inference complete.\n");
    fprintf("Results saved in: %s\n", outputDir);

end