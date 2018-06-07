function [M, pDgHI, interventions, uniqueInt] = pDH_bnet(bnet, examples, prior)

for i = 1 : length(bnet)
    done = zeros(1,size(examples,1));

    for j = 1 : size(examples,1)
        
        if done(j)==0
            intervene = find(examples(j,end/2+1:end) < 1);
            b = bayes_net_state_set(bnet{i}.node, examples(j,intervene), 1);

            % consider all other possible outcomes
            obsvd = find(examples(j,end/2+1:end) > 0);
            intervened = find(examples(j,end/2+1:end) < 1);
            tmp = zeros(1,length(obsvd));
            
            if ~isempty(obsvd)
            scores=[]; exNum=[];
            for k = 1 : 2^length(obsvd)
                % set state
                b = bayes_net_state_set(b, examples(j,obsvd), tmp);

                % score this state
                scores(k) = bayes_net_score_intervention(b, intervened);
                
                % find the example corresponding to this state
                tmpEx=examples(j,:); tmpEx(obsvd+size(examples,2)/2)= tmp+1;
                exNum(k) = find(sum(examples==repmat(tmpEx,size(examples,1),1),2)==size(examples,2));

                tmp = increment_variable_base_vec(tmp, ones(1,length(obsvd))*2);
                
            end
            
            else
                scores = 1; % if all variables are set, probability is 1
                exNum = j;
            end
            
            % record these for later marginalization
            for k = 1 : length(exNum)
                exSets{exNum(k)} = exNum;
            end
            
            % update scores for all entries
            M(exNum,i) = scores; 
            done(exNum) = 1;
            
        end
    end
    
end
pDgHI = M;
% M, aka p(d|h,i) does not sum to 1, sums to 27 because there are 27 possible
% interventions

% marginalize over possible observed data given h,i
done = zeros(1,size(M,1)); interventions = zeros(1,size(M,1)); ctr = 1;
for i = 1 : size(M,1)
    
    if done(i) ==0
        interventions(exSets{i}) = ctr;
        
        % computing \sum_d \frac{p(d|h,i)*p(h)}{\sum_h p(d|h,i)*p(h)},
        % assume p(i|h) is uniform 
        denom = repmat(sum(M(exSets{i},:).*repmat(prior,length(exSets{i}),1),2),1,size(M,2));
        z = denom==0; % deal with zeros
        tmp = M(exSets{i},:).*repmat(prior,length(exSets{i}),1)./ denom;
        tmp(z) = 0; tmp = sum(tmp,1);
        tmp = normalize(tmp); % sum over h equals 1
        M(exSets{i},:) = repmat(tmp, length(exSets{i}), 1);
        done(exSets{i}) = 1;
        ctr = ctr + 1;
    end
    
end

[a,uniqueInt] = unique(interventions);
new = normalize(M(uniqueInt,:),1);
M = new(interventions,:); % p(i|h) with copies of this for different data

