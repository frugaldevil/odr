function [] = downgrade_digitStruct()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load('train/digitStruct.mat')
save('train/digitStruct.mat', 'digitStruct', '-v6')

load('test/digitStruct.mat')
save('test/digitStruct.mat', 'digitStruct', '-v6')

load('extra/digitStruct.mat')
save('extra/digitStruct.mat', 'digitStruct', '-v6')
end

