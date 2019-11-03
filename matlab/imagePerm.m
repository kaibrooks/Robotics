% Image Perm
% Kai Brooks
% github.com/kaibrooks
% 2019
%
% takes an image and makes a bunch of permutations of it for training an image recognition algorithm

clc; close all; clear all; rng('shuffle');

% user settings -----------------------------------------------------------

makeImages = 20; % (20) images to make
repeatProb = 0.4; % (0.4) probability an image goes back through filtering again
maxAngle = 45; % (45) max angle rotations will make
fuzz = 0.1; % (0.1) fuzz in noise

randomizeFilenames = 0; % randomizes filename prefixes so sequential runs (probably) don't overwrite each other
deleteExistingFiles = 0; % deletes previous output before saving new run
WEIRDNESS = 1; % (0.3-1.0) beeeeewaaaaaaaaaree


% image to load in
cat = imread('images/cat.jpg');

% other vars (no touch) ---------------------------------------------------

rotated = 0;
flipped = 0;
filts = 0;
cont = '';

prefix = randi([100 999]); % used if randomizedFilenames            

% go ----------------------------------------------------------------------


theFiles = dir(fullfile('images/output/', '*.jpg'));

if deleteExistingFiles % delete previous files
    for k = 1 : length(theFiles)
        baseFileName = theFiles(k).name;
        fullFileName = fullfile('images/output/', baseFileName);
        fprintf(1, 'Deleting %s\n', fullFileName);
        delete(fullFileName);
    end
end

% check if data exists and ask to overwrite

if size(theFiles) > 0;
    cont = input('Files already exist and may be overwritten. Y to continue: ','s');
    if upper(cont) ~= "Y"
        fprintf('End\n')
        return
    end 
end

fprintf('Starting...\n');

im = cat; % set to base image

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
            temp = imnoise(im,'gaussian',0.0,adjFactor*fuzz*WEIRDNESS);
            
        case 5 % change contrast
            temp = imadjust(im,[.1 .2 0; .8*WEIRDNESS .9*WEIRDNESS 1],[]);    
            
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
        if randomizeFilenames
            padded = sprintf( '%i%03d',prefix,i); % prefix and add trailing zeroes
        else
        padded = sprintf( '%03d',i); % add padded zeroes to filename
        end
        
        
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

fprintf('Done\n');
