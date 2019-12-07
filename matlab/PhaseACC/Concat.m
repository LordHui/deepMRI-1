function [data] = Concat(x)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
ll = size(x);
ll = [ll 2];
data = zeros(ll);
x = x ./ max(abs(x(:)));
data(:,:,:,1) = real(x);
data(:,:,:,2) = imag(x);
end
