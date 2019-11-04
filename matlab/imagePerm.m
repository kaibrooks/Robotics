% Image Perm
% Kai Brooks
% github.com/kaibrooks
% 2019
% MATLAB R2018a
%
% takes an image and makes a bunch of permutations of it for training an image recognition algorithm
%
% folder structure must be:
% (base dir)/images/training    for the input (training data) files
% (base dir)/images/output      for the output (permutated) files
% images must be .jpg

clc; close all; clear all; rng('shuffle');

% user settings -----------------------------------------------------------

makeImages = 20; % (20) images to make
repeatProb = 0.4; % (0.4) probability an image goes back through filtering again
maxAngle = 45; % (45) max angle rotations will make
fuzz = 0.1; % (0.1) fuzz in noise

deleteExistingFiles = 1; % deletes previous output before saving new run

% other vars (no touch) ---------------------------------------------------

rotated = 0;
flipped = 0;
filts = 0;
cont = '';

% go ----------------------------------------------------------------------

% check for older data
oldFiles = dir(fullfile('images/output/', '*.jpg')); % existing output from previous runs
if deleteExistingFiles % delete previous files
    for k = 1 : length(oldFiles)
        baseFileName = oldFiles(k).name;
        fullFileName = fullfile('images/output/', baseFileName);
        fprintf(1, 'Deleting %s\n', fullFileName);
        delete(fullFileName);
    end
    oldFiles = dir(fullfile('images/output/', '*.jpg'));
end

% check if data exists and ask to overwrite
if size(oldFiles) > 0;
    cont = input('Files already exist and may be overwritten. Y to continue: ','s');
    if upper(cont) ~= "Y"
        fprintf('End\n')
        return
    end
end

% get contents of training folder
getImages = dir(fullfile('images/training/', '*.jpg'));
getTxts = dir(fullfile('images/training/', '*.txt'));

% end if training folder is empty or unreadable
if length(getImages) == 0
    fprintf('No .jpg images in images/training/\nEnd\n')
    return
end

% warn if there isn't a matching txt for each image
if length(getImages) ~= length(getTxts)
    prompt = sprintf('Warning: Unequal images and texts in training (%i images, %i txts). Y to continue: ',length(getImages),length(getTxts));
    cont = input(prompt,'s');
    if upper(cont) ~= "Y"
        fprintf('End\n')
        return
    end
end

% actual loop start
fprintf('Starting...\n');
for j = 1:length(getImages)
    
    % get image
    im = imread(fullfile('images/training/', getImages(j).name));
    
    % get filename of image to prefix output permutations
    outputPrefix = erase(getImages(j).name,'.jpg');
    
    while i < makeImages
        adjFactor = rand();
        alg = randi(7); % keep between 1 and 5 to disallow black and white
        filts = filts + 1;
        
        switch alg
            
            case 1 % flip
                if flipped
                    continue
                end
                temp = flipdim(im, 2);           % horizontal flip
                
            case 2 % flip again for more probability
                if flipped
                    continue
                end
                temp = flipdim(im, 2);           % horizontal flip
                
            case 3 % rotate
                if rotated
                    continue
                end
                temp = imrotate(im,randi([1 maxAngle]),'crop');
                rotated = 1;
                
            case 4 % make fuzzy
                temp = imnoise(im,'gaussian',0.0,adjFactor*fuzz);
                
            case 5 % change contrast
                temp = imadjust(im,[.1 .2 0; .8 .9 1],[]);
                
            case 6 % make b&w and increase contrast
                temp = imadjust(rgb2gray(im),[0.1 0.9],[]);
                
            case 7 % denoise (must be b&w)
                amount = randi(4)+2;
                temp = wiener2(rgb2gray(im),[amount amount]);
                
            case 8 % adjust HSV
                adjFactor = adjFactor;
                temp = rgb2hsv(im);
                temp(:, :, 2) = temp(:, :, 2) * adjFactor;
                
        end % switch
        
        imshow(temp);
        f=getframe;
        imwrite(f.cdata,'images/temp.png');
        
        % write image and move on to next
        if rand() > repeatProb
            
            % write file
            
            padded = sprintf( '%s_%03d',outputPrefix,i); % prefix and add trailing zeroes
            
            filename = sprintf('images/output/%s.jpg', padded);
            imwrite(f.cdata, filename);
            
            fprintf("Image %s created with %i filters\n",padded, filts)
            
            % reset for next run
            im = cat; % rest to base image
            rotated = 0;
            flipped = 0;
            filts = 0;
            i = i + 1;
        else
            %fprintf("Looping after applying %i\n", alg)
            im = imread('images/temp.png');
        end % if termChance
        
    end % 1:makeImages
    fprintf('Image %s finished\n',outputPrefix)
    i = 0;
end % 1:length(getImages)

fprintf('Done\n');