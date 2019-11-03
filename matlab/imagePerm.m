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

makeImages = 50;
termChance = 0.5; % 0 for ~ ~ ~ w a r h o l m o d e ~ ~ ~

cat = imread('images/cat.jpg');
I = imread('images/cat.jpg');
%imshow(cat)

im = cat;

% get details about it
%imhist(cat)
[height, width, colorSpace] = size(im);
imCenter = [width/2 height/2];

while i < makeImages
    adjFactor = rand();
    alg = randi(4);
    
    switch alg
        case 8 % make b&w
            temp = rgb2gray(im);
            
        case 2 % make fuzzy
            temp = imnoise(im,'gaussian',0.0,adjFactor/2);
      
        case 1 % flip
            temp = flipdim(im, 2);           % horizontal flip
            
        case 3 % change contrast
            temp = imadjust(im,[.2 .3 0; .6 .7 1],[]);
   
        case 4 % rotate
            temp = imrotate(im,randi([1 90]));
            
              
        case 8 % adjust HSV
            adjFactor = adjFactor;
            temp = rgb2hsv(im);
            temp(:, :, 2) = temp(:, :, 2) * adjFactor;
            
        case 8 % denoise (must be b&w)
            temp = wiener2(rgb2gray(imnoise(im,'gaussian',0.0,adjFactor/2)),[5 5]);
            
    end % switch
    
    % crop it
    
    [newH, newW, colorSpace] = size(temp); % get width / height
    imCenter = [newW/2 newH/2];
    cropRect = [(newW/2)-(width/2) newH/2-(height/2) width height]; % xmin ymin width height

    
    
    temp = imcrop(temp,cropRect);
    %temp = centerCropWindow2d(size(I),targetSize);
    
    imshow(temp);
    saveas(gcf,'images/temp.jpg', 'jpg')
    
    % write image and move on to next
    if rand() < termChance
        saveas(gcf,'images/1.jpg', 'jpg')        
        i = i + 1;
        fprintf("Image %i created using %i\n",i, alg)
        im = cat;
    else
       fprintf("Looping after %i\n", alg)
       im = imread('images/temp.jpg'); 
    end
    
    
    
end % 1:makeImages
