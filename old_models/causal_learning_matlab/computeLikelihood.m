function M = computeLikelihood(M, pDgHI, uniqueInt, interventions, prior, rho, nIter)

crit = 0.00001;

M1 = zeros(size(M));

while sum(sum(abs(M-M1))) > crit
%for i = 1 : nIter
    M1 = M;
    M = f(M, pDgHI, uniqueInt, interventions, prior, rho); % p(i|h)
    new = normalize(M(uniqueInt,:),1);
    M = new(interventions,:);
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function M = f(M,pDgHI, uniqueInt, interventions, prior, rho)

for i = 1 : max(interventions)
    t = interventions == i;
    % computing \sum_d \frac{p(i|h)*p(d|h,i)*p(h)}{\sum_h
    % p(i|h)*p(d|h,i)*p(h)}
    denom = repmat(sum(M(t,:).*pDgHI(t,:).*repmat(prior,sum(t),1),2),1,size(M,2));
    z = denom==0; % deal with zeros
    tmp = M(t,:).*repmat(prior,sum(t),1)./ denom;
    tmp(z) = 0; tmp = sum(tmp,1);
    tmp = normalize(tmp); % sum over h equals 1
    M(t,:) = repmat(tmp, sum(t), 1);
end

M = normalize(M.^rho,2);