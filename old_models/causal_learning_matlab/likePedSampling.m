function M = likePedSampling(conceptPriors,examples,M,rho,extension)

% returns M, a matrix of likelihoods of examples (rows) 
% for each hypothesis (columns)

crit = 0.00001; % this is the stopping criterion for our fixed point

% repmat pH
pH = repmat(conceptPriors', size(M,1), 1);


% initialize Mprime, we use this to check if converged
Mprime = ones(size(M));

% iterate until criterion
while sum(sum(abs(Mprime-M))) > crit 
  Mprime = M;
  M = f(M,pH,rho,extension,examples); % compute p(h|d)
  M = round(normalize(M,1)*10^8) ./ 10^8; % p(d|h)
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function M = f(M,pH,rho,extension,examples1)

% if ext == 1: likelihood \propto \sum_y \sum_h p(y \in C|h)*p(h|d)

% TESTING, condiion on #pos,neg
%x = sum(examples1(:,end/2+1:end),2);
%ux = unique(x);
%for i = 1 : length(ux)
%   M(x==ux(i),:) = normalize(M(x==ux(i),:),1);
%end
% END TESTING

num = pH.*M; % p(h)*p(d|h)
denom = sum(pH.*M,2); % \sum_h p(h)*p(d|h)
% deal with zeros
%denom(denom == 0) = 10^-10;
denom = repmat(denom, 1, size(pH,2));

M(denom>0) = (num(denom>0) ./ denom(denom>0));
M(denom==0)=1; M(num==0)=0; 

M = M.^rho;

