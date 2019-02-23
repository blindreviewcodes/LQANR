function [G,a] = gsp_admm(P,K,m,alfa,rou,r)
%%min(a,B,C) sum a_i*||P_i-B*C_i||
% B is integer, eg. -1,0,1
nodenum = size(P,1);
rng('default');
%B=randn(nodenum,m);  %-1£¬1
B = 2*rand(nodenum,m)-1;  %-1,0,1
%B = 4*rand(nodenum,m)-2; %-2,-1,0,1,2
%B = 8*rand(nodenum,m)-4; %-4£¬-2,-1,0,1,2£¬4
%G=sign(B);
G = round(B);
lamda=ones(nodenum,m);
    maxiter = 2;
    iter_in = 10;
    ABSTOL   = 1e-3;
    RELTOL   = 1e-2;
    tol = 1e-4;
    a=ones(K,1)/K;
  %  a=[0.36 0.48 0.16];
    C=zeros(m,size(P,2),K);
%% solve optimization function by fixing one variable each time
for i = 1:maxiter
     N1=zeros(m,m);
     N2=zeros(m,size(P,2));
    for j=1:K
        C(:,:,j) = (a(j)*G'*G+alfa*eye(m))\(a(j)*G'*P(:,:,j));
    end
    r_norm = zeros(iter_in,1);
    s_norm = zeros(iter_in,1);
    for j = 1:iter_in
        N3=zeros(nodenum,m);
        N4=zeros(m,m);
        for k=1:K
            N4=N4+a(k)*C(:,:,k)*C(:,:,k)';
            N3=N3+a(k)*P(:,:,k)*C(:,:,k)';
        end
    B=(N3+rou*G-rou*lamda)/(N4+rou*eye(m));
    Gold = G;
    
   % G = sign(B+lamda);  % G:-1,1
    G = round(B + lamda);  %update indicator function
    G(G>1)=1;
    G(G<-1)=-1;
%     G(G>2)=2;
%     G(G<-2)=-2;
 
%      G=B+lamda;   % G=-4,-2,-1,0,1,2
%       G(G>=3)=4;
%      G(G<=-3)=-4;
%      G(G>2 & G<3)=2;
%      G(G>-3 & G<-2)=-2;
%      G=round(G);
     
    lamda = lamda + B - G;
    
    r_norm(j)  = norm(B - G);
    s_norm(j)  = norm(-rou*(G - Gold));
    eps_pri = ABSTOL*nodenum + RELTOL*max(norm(B), norm(-G));
     eps_dual= ABSTOL*m + RELTOL*norm(rou*lamda);
      if (r_norm(j) < eps_pri && s_norm(j) < eps_dual)
         break;
     end
    end
     M_norm=zeros(1,K);
    M_sum=0;
    for j=1:K
        M=P(:,:,j)- G*C(:,:,j);
        M_norm(j)=norm(M(:),2);
        M_sum=M_sum+(1/M_norm(j))^(2/(r-1));
    end
    for j=1:K
        a(j)= (1/M_norm(j))^(2/(r-1))/M_sum;
    end
end