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
rand_bin = round(rand(1,8));


% mix

% rebreed

% display


% calculate bf state space
for i=2:3
    x = i;
    y = i;
    
    z = x*y;
    
    motion(i) = z^2%3*(x*y - 2);
    n(i) = i;
    
    plotex(i) = exp(z);
    plotx2(i) = z^2;
    plotx3(i) = z^3;
    plotxbang(i) = factorial(z);
end

motion % note 1 is invalid, also total state space of sim is tripled (one for each bot)


hold on
grid on
plot(motion,'LineWidth',2)
plot(plotx2)
plot(plotx3)
plot(plotex)
%plot(plotxbang)


legend('motion','x^2','x^3','e^x','n!')