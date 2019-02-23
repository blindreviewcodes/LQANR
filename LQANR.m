%% Low-Bit Quantization for Attributed Network Representation Learning
load cora.mat %wiki.mat %blogcatalog.mat %cora.mat citeseer.mat
Z = diag(sum(attributes.^2,2).^-.5); % temporary value
Z(isinf(Z)) = 0; % temporary value
X = Z'*attributes;
clear Z;
nodenum=size(network,2);
A=network;
A = A+eye(nodenum);
L=diag(sum(A,1).^(-1))*A;
%% Calulate P
K=5;
P=zeros(nodenum,size(X,2),K);
P(:,:,1)=L*X;
for i=2:K
    P(:,:,i)=L*P(:,:,i-1);
end
%% Calculate B
B=gsp_admm(P,5,100,0.0001,0.0003,1.4);
%% Classfication
train_ratio = 0.1;
C = 5;
numOfGroup = 7; %citeseer:6  wiki:19  blog:6   cora:7
[F1macro2,F1micro2] = svmTest(B,labels,numOfGroup,train_ratio,C)
