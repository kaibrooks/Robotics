% Kai Brooks
% github.com/kaibrooks
% Needs version 2019b

clc
close all
clear all
rng('shuffle')

% get some files 
if ~exist('flower_photos','dir') % but only if we haven't already
    fprintf('Getting test images...\n')
    
%     url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';
%     downloadFolder = tempdir;
%     filename = fullfile(downloadFolder,'flower_dataset.tgz');
%     
%     imageFolder = fullfile(downloadFolder,'flower_photos');
%     if ~exist(imageFolder,'dir')
%         disp('Downloading Flower Dataset (218 MB)...')
%         websave(filename,url);
%         disp('Unzipping...')
%         untar(filename,downloadFolder)
%     end
end

% locate the folder path
datasetFolder = fullfile('flower_photos','sunflowers')

% Create an image datastore containing the photos of sunflowers only.
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Augment the data to include random horizontal flipping, scaling, and resize the images to have size 64-by-64.
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandScale',[1 2]);
augimds = augmentedImageDatastore([64 64],imds,'DataAugmentation',augmenter);












