%% Divide Network
load('wiki.mat'); 
nodenum=size(network,2);
Z = diag(sum(attributes.^2,2).^-.5); % temporary value
Z(isinf(Z)) = 0; % temporary value
X = Z'*attributes;
clear Z;
net = network;
ratioTrain = 0.9; 
[train, test] = DivideNet(net,ratioTrain);              
train = sparse(train); 
test = sparse(test);
train = spones(train + train'); 
test = spones(test+test');           
%% 
A = train;
A = A+eye(nodenum);
L=diag(sum(A,1).^(-1))*A;
K=3;
P=zeros(nodenum,size(X,2),K);
P(:,:,1)=X;
for i=2:K
    P(:,:,i)=L*P(:,:,i-1);
end
% [U,S1] = svds(P, 200);
% P = U*S1;
% B=dcc(P',100,0.001);
%G=gsp_admm5(P,5,100,0.0001,0.0003,1.4,labels);  %cora
%G=gsp_admm5(P,6,100,0.0001,0.0003,1.5,labels);  %citeseer
%G=gsp_admm5(P,3,100,0.0001,0.0004,1.1,labels);  %wiki
 G=gsp_admm5(P,3,100,0.0001,0.0004,5,labels);   %Blog
%%
S = G*G';
n = 10000;
CalcAUC( train, test, S, n )