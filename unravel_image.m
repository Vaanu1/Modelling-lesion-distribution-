function [urimage,sz] = unravel_image(J,k)

sz=size(J);
n=length(sz);

%dimensions other k-th dimension, along which convolution will happen:
otherdims = 1:n; otherdims(k)=[];

% permute order: place k-th dimension as 1st, followed by all others:
indx1=[k otherdims];

%perform convolution along k-th dimension:
%
%1. permute dimensions to place k-th dimension as 1st
J = permute(J,indx1);
%2. create a 2D array (i.e. "stack" all other dimensions, other than
%k-th:
J = reshape(J,sz(k),prod(sz(otherdims)));

urimage = J;

