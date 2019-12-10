%% Graduate_change_mask
%% There are 5 layers: 0-30;31-50;51-80;81-100;101-128
%% The downsampling ratio is dependent on the threshould of each circle layer

close all; clear all;

N=256;
Mask=zeros(N,N);
[columnsInImage rowsInImage] = meshgrid(1:N,1:N);

%% Set the original p[oint
centerX = N/2;
centerY = N/2;

%% Layer1: 0-30 Fully sampling
radius1 = 30;
Mask1 = (rowsInImage - centerY).^2 ...
    + (columnsInImage - centerX).^2 <= radius1.^2;


%% Layer2: 31-50
radius2=50;
mask=normrnd(0,1,[N,N]);
threshold=0;
Mask=zeros(N,N);

for i=1:N
    for j=1:N
        if mask(i,j)>threshold
            Mask(i,j)=1;
        end
    end
end
Mask=Mask+Mask1;

for i=1:N
    for j=1:N
        if ((i-128).^2+(j-128).^2-radius2.^2>0)
            Mask(i,j)=0;
        end
    end
end


%% Layer3: 51-80
radius3=80;
mask=normrnd(0,1,[N,N]);
threshold=0.5;
Mask2=zeros(N,N);
for i=1:N
    for j=1:N
        if mask(i,j)>threshold
            Mask2(i,j)=1;
        end
    end
end
Mask=Mask+Mask2;


for i=1:N
    for j=1:N
        if ((i-128).^2+(j-128).^2-radius3.^2>0)
            Mask(i,j)=0;
        end
    end
end


%% Layer4: 81-100
radius4=100;
mask=normrnd(0,1,[N,N]);
threshold=1.1;
Mask3=zeros(N,N);
for i=1:N
    for j=1:N
        if mask(i,j)>threshold
            Mask3(i,j)=1;
        end
    end
end
Mask=Mask+Mask3;

for i=1:N
    for j=1:N
        if ((i-128).^2+(j-128).^2-radius4.^2>0)
            Mask(i,j)=0;
        end
    end
end
figure,imshow(Mask);

%% Layer5: 100-128
radius5=128;
mask=normrnd(0,1,[N,N]);
threshold=1.55;
Mask4=zeros(N,N);
n1=0;
for i=1:N
    for j=1:N
        if mask(i,j)>threshold
            Mask4(i,j)=1;
            n1=n1+1;
        end
    end
end
Mask=Mask+Mask4;

for i=1:N
    for j=1:N
        if ((i-128).^2+(j-128).^2-radius5.^2>0)
            Mask(i,j)=0;
        end
    end
end



%% Layer6: >128
radius5=128;
mask=normrnd(0,1,[N,N]);
threshold=2.0;
Mask5=zeros(N,N);
n1=0;
for i=1:N
    for j=1:N
        if mask(i,j)>threshold
            Mask5(i,j)=1;
            n1=n1+1;
        end
    end
end
Mask=Mask+Mask5;


%% Calculating the undersampling ratio
count=0;
for i=1:256
    for j=1:256
        if Mask(i,j)>0;
            Mask(i,j)=1;
        end
    end
end

for i=1:256
    for j=1:256
        if Mask(i,j)==1;
            count=count+1;
        end
    end
end

figure,imshow(Mask);
ratio=(256*256)/count

