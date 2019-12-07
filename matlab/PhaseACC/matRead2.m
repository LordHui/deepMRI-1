function data = matRead2(filename)
% data = matRead(filename) reads the image data in the MAT-file filename

inp = load(filename);
f = fields(inp);
data = inp.(f{1});
ll = size(data);
ll = [ll 2];
temp = zeros(ll);
temp(:,:,:,1) = real(data);
temp(:,:,:,2) = imag(data);
data = temp; 
end