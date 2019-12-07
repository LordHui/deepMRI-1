function [x_sub] = Subsampling(x, Mask)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    x_sub = x .* Mask; 
end

