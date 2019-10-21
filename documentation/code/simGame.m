% Robot cop/thief model
% Kai Brooks
% github.com/kaibrooks
% 2019
%
% Does a thing
%

% 3 axes of motion
% action space for agent:
% stay, left, right, up/down
% action space = 4
%
% state actions for agent:
% independent
% dependent on 1 robber
% dependent on 2 robber
% dependent on both
%
% total state space = 16


% Init
clc
close all
clear all
format
rng('shuffle')


% generate initial chromosome

lengthX = 6;
lengthY = 6;
maxPop = 10;

% generated internal vars
board = zeros(lengthX);
chromLength = lengthX * lengthY * 3; % size of the board, *3 for 3 bits

% generate chromosome
for n = 1:maxPop
    population(n,:) = round(rand(1,chromLength));
end




% mix

board(3,1) = 1; % starting position
lastPosY = 3;
lastPosX = 1;

% get nearby spaces
nextY = [lastPosY-1 lastPosY+1];
nextX = [lastPosX-1 lastPosX+1];

xw5x
% zero moves to large (off the board)
nextY(nextY>=lengthY) = 0;
nextY(nextY<=1) = 0;

% zero moves too small (off the board)
nextX(nextX>=lengthX) = 0;
nextX(nextX<=1) = 0;


% evaluate population
for i = 1:maxPop
    
    
    
end % 1:maxPop


% rebreed

% display