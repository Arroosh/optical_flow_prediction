function [rectI, finalMask] = applyH(img, H, mask)
  tform = projective2d(H');
  Rout = imref2d(size(img));
  [rectI] =imwarp(img,tform, 'OutputView', Rout);
  [finalMask] =imwarp(mask,tform, 'OutputView', Rout);

  % imshow(rectI);
end