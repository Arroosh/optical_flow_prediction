
N2 = [];
B = [];
theDim = sqrt(length(A));
for i = 1:theDim
    i
for j = 1:theDim
B(i,j) = A((i-1)*theDim + j);
end
end
A = B;
NvX = codebook(A(:), 1);
NvY = codebook(A(:), 2);
N2(:,:,1) = reshape(NvX, [theDim theDim]);
N2(:,:,2) = reshape(NvY, [theDim theDim]);
rgbOut = uint8(flowToColor(N2));
imshow(rgbOut)