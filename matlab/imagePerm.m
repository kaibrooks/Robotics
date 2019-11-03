% Image Perm
% Kai Brooks
% github.com/kaibrooks
% 2019
%
% takes an image and makes a bunch of permutations of it for training an image recognition algorithm

clc
close all
clear all
rng('shuffle')

% user settings ------------------------------------------------

makeImages = 20; % (20) images to make
termChance = 0.4; % (0.4) 0 for ~ ~ ~ w a r h o l m o d e ~ ~ ~
maxAngle = 45; % (45) max angle rotations will make
fuzz = 0.1; % (0.1) fuzz in noise

WEIRDNESS = 1; % (0.3-1.0) beeeeewaaaaaaaaaree

% image to load in
cat = imread('images/cat.jpg');

% other vars ---------------------------------------------

rotated = 0;
flipped = 0;
filts = 0;

% go ----------------------------------------------------------------

im = cat;

% get details about it
%imhist(cat)
[height, width, colorSpace] = size(im);
imCenter = [width/2 height/2];

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
    
    % crop it
    [newH, newW, colorSpace] = size(temp); % get width / height
    imXY = [(newW/2)-(width/2) newH/2-(height/2)];
    cropRect = [[imXY] width height]; % xmin ymin width height
    
    %temp = imcrop(temp,cropRect);
    
    imshow(temp);
    f=getframe;
    imwrite(f.cdata,'images/temp.png');
    %saveas(gcf,'images/temp.jpg', 'jpg')
    
    % write image and move on to next
    if rand() < termChance
        
        % write file
        f=getframe;
        padded = sprintf( '%03d',i); % add trailing zeroes to filename
        filename = sprintf('images/output/%s.jpg', padded);
        imwrite(f.cdata, filename);
        
        fprintf("Image %s created with %i filters\n",padded, filts)
        
        % reset for next run
        filts = 0;
        i = i + 1;
        im = cat;
        rotated = 0;
        flipped = 0;
    else
        %fprintf("Looping after applying %i\n", alg)
        im = imread('images/temp.png');
    end
    
    
    
end % 1:makeImages

fprintf('Done\n');
