%% data preprocessing
volReader = @(x) matRead(x);
labels = imageDatastore('../**/kspace.mat', ...
            'FileExtensions', '.mat', 'ReadFcn', volReader);
%% get the number of the files. 
NumFiles = length(inputs.Files);

%% preprocessing 1
for m = 1 : NumFiles
    temp = read(labels);  % temp: kspace data, size [nx, ny, nz, 8]
    for n = 1 : 8
        k_full = temp(:,:,:,n);
        temp2 = (m - 1) * 8 + n; 
        img_full = fftn(fftshift(k_full));
        %% normalisation
        k_full = k_full ./ max(abs(k_full(:)));
        img_full = img_full ./ max(abs(img_full(:)));
        %% save files. 
        file_name = strcat('./k_labels/k_full_', num2str(temp2), '.mat');     
        save(file_name, 'k_full');
        file_name2 = strcat('./img_labels/img_full_', num2str(temp2), '.mat');     
        save(file_name2, 'k_full');
    end
end 
