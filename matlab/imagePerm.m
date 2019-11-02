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

makeImages = 30;
termChance = 0.5;


cat = imread('images/cat.jpg');
I = imread('images/cat.jpg');
%imshow(cat)

% get details about it
%imhist(cat)

im = cat;

while i < makeImages
    adjFactor = rand();
    alg = randi(7);
    
    switch alg;
        case 8 % make b&w
            temp = rgb2gray(im);
            
        case 2 % make fuzzy
            
            temp = imnoise(im,'gaussian',0.0,adjFactor/2);
            
        case 8 % adjust HSV
            adjFactor = adjFactor;
            temp = rgb2hsv(im);
            temp(:, :, 2) = temp(:, :, 2) * adjFactor;
            
        case 4 % denoise (must be b&w)
            temp = wiener2(rgb2gray(imnoise(im,'gaussian',0.0,adjFactor/2)),[5 5]);
            
        case 5 % flip
            temp = flipdim(im, 2);           % horizontal flip
            
        case 6 % change contrast
            temp = imadjust(im,[.2 .3 0; .6 .7 1],[]);
   
        case 7 % rotate
            temp = imrotate(im,randi([1 90]),'crop');
            
    end % switch
    
    imshow(temp);
    saveas(gcf,'images/temp.jpg', 'jpg')
    
    % write image and move on to next
    if rand() < termChance
        saveas(gcf,'images/1.jpg', 'jpg')        
        i = i + 1;
        fprintf("Image %i created using %i\n",i, alg)
    else
       fprintf("Looping after %i\n", alg)
       im = imread('images/temp.jpg'); 
    end
    
    
    
end % 1:makeImages
