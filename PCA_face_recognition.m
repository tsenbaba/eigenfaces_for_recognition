%This code is implementation for the article "Eigenfaces for Recognition. Matthew Turk and Alex Pentland"
clear, clc
% implement face regocnition using PCA
K= 9; % number of eigen values to keep
M= 32; N=32; %picture dimention M x N

%Image loading and data for training
%convert each picture to double as data type
%create X matrix where each column represent a picture 
imagefiles = dir('*.TIF');
n=length(imagefiles); %number of picture for train
X = [];

for i=1:n
imagename = imagefiles(i).name;
tempimage = imread(imagename);
tempimage  =double(tempimage); %change the data type to double
X(:,i) = reshape(tempimage, (M*N), 1);
end
size(X);

mu = mean(X,2); %compute the mean of images as vector
m = repmat(mu, 1, 35); %expand mean vector to the matrix. each column is mean vector 'mu'
cov = (X - m)*(X - m)'; %covariance matrix
[EvecX, EvalX] = eig(cov); % get eigenvalues and eigenvectors from covariance matrix
[EvalSorted, index] = sort(diag(EvalX), 'descend'); %Sort eigenvalues
SortedEval = EvecX(:, index);

Ppca = SortedEval(:,1:K); %Keep 'k' number of eigenvalues for further calculation

%Known faces Database Construction
FAdir = ('/Users/tsenbaba/Documents/MATLAB/FA');
fa_image = dir(fullfile(FAdir,'*.TIF'));
fa =length(fa_image); %fa = 12
X_fa = [];
for i=1:fa
imagename = fa_image(i).name;
tempimage = imread(imagename);
tempimage  =double(tempimage); %change the data type to double
X_fa(:,i) = reshape(tempimage , (M*N), 1) - mu;
end
db_fa = Ppca' * X_fa; ) %size(db_fa) =  (k x 12)


%Test Faces Database Construction
FBdir = ('/Users/tsenbaba/Documents/MATLAB/FB');
fb_image = dir(fullfile(FBdir,'*.TIF'));
fb =length(fb_image); %fb=23
X_fb = [];

for i=1:fb
imagename = fb_image(i).name;
tempimage = imread(imagename);
tempimage  =double(tempimage); %change the data type to double
X_fb(:,i) = reshape(tempimage , (M*N), 1) - mu;
end
db_fb = (Ppca' * X_fb)'; %size(db_fb) =  (23 x k)

% Recognition by nearest neighbor classification
% Try to match pictures from Test Faces Database to Known faces Database 
count=0; %counts the number of pictures that match correctly
for i=1:fb
    %calculate euclidean distance
    [value,index] = min(sqrt(sum(((db_fb(i,:) - db_fa') .^ 2),2))); 
    %check if the train image match the known images
    if fb_image(i).name(7:11) == fa_image(index).name(7:11)
        count= count+ 1;
    end
end

%ratio shows correctly matched pictures over total number of test pictures
%pictures which are tested
accuracy = (count/fb) * 100
