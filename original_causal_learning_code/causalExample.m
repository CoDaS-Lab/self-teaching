function [posterior, likelihoods, examples, pDgHI] = causalExample(models, numExSets)

% this version uses causal models where only intervention is on.

% NOTE: models chooses which causal models we are considering!!!
%     : numExSets tells how many sets of interventions teacher gave.

% Settings for this run
numNodes = 3; 
numVals = 3; % 3 total {interveneOn, observedOff, observedOn}

% how helpful teacher is
rho = 1;
nIter = 0; % to turn on go in code for compute likelihood
nObs = 3;

% parameters for causal model (noisy-or)
b = 0.01; t = .99; % FIXME: differentiate between b and root nodes?

[concepts bnet] = makeConcepts(b, t);
% this is to select only some
% 1:3 common cause
% 4:6 common effect
% 7:12 chain
% 13:18 chain with loop
% 19:24 single cause
% 25 no cause
bnet = bnet(models);
concepts = concepts(models,:);

prior{1} = normalize(ones(1,length(bnet)));

% deal with possible sets of interventions
% NOTE: this generates all possible intervention-observation sets
intEx = [];
nck = nchoosek(nObs,numNodes);
examples = zeros(nck*numVals.^numNodes, numNodes.*2);
examples(1:nck, 1:numNodes) = nchoosek(1:nObs, numNodes);
temp = zeros(1,numNodes);
base = ones(1,numNodes)*numVals;
sc = size(examples,2) ./ 2 + 1;
for j = 1 : max([numVals.^numNodes,2])
  sr = nck*(j-1)+1; fr = nck*j;
  examples(sr:fr,sc:end) = repmat(temp,nck,1);
  examples(sr:fr,1:sc-1) = examples(1:nck, 1:sc-1);
  temp = increment_variable_base_vec(temp, base);
end

% only examples with one intervention
examples = examples(sum(examples(:,end/2+1:end)==0,2)==1,:);

% FIXME: add in greedy parameter
[M, pDgHI, interventions, uniqueInt] = pDH_bnet(bnet, examples, prior{1});
M = computeLikelihood(M, pDgHI, uniqueInt, interventions, prior{1}, rho, nIter);
likelihoods{1} = M;
posterior{1} = normalize(likelihoods{1} .* pDgHI.* repmat(prior{1}, size(examples,1), 1),2);

if numExSets == 1
    return;
else
    %keyboard
end

% these are conditional on first set (without observations)
like2 = cell(1,3);
post2 = cell(1,3);
prior2 = cell(1,3);

% for each intervention, calculate beliefs, then redo choice
for i = 1 : 3
    prior2{i} = normalize(normalize(sum(posterior{1}(interventions==i,:),1)).^rho);
    [M, pDgHI, interventions, uniqueInt] = pDH_bnet(bnet, examples, prior2{i});
    like2{i} = computeLikelihood(M, pDgHI, uniqueInt, interventions, prior2{i}, rho, nIter);
    post2{i} = normalize(like2{i} .* pDgHI.* repmat(prior2{i}, size(examples,1), 1),2);
end

save causalPreds2 likelihoods posterior examples bnet concepts like2 post2 prior2 prior

% % NOTE: the following is for calculating order effects
% % if want computation for more than one example, must now choose which
% % example was seen then compute rest conditioned on that.
% ex2 = 1:4; % choose all with the same intervention if teacher does not get to see outcome
% 
% % FIXME: JOT DOWN WHICH EXAMPLES ARE GIVEN FIRST IN EXPERIMENT.
% 
% for i = 2: numExSets
%    prior{i} = normalize(sum(posterior{i-1}(ex2,:),1));
%    M = computeLikelihood(M, pDgHI, uniqueInt, interventions, prior{i});
%    likelihoods{i} = M;
%    posterior{i} = normalize(likelihoods{i} .* pDgHI.* repmat(prior{i}, size(examples,1), 1),2);
% end
% 
% keyboard
% % FIXME: SAVE!
