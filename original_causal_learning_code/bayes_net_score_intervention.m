function prob = bayes_net_score_intervention(bnet, var)
% var indicates intervened variables
% cycle through topological order

prob = 1;

for i = 1 : length(bnet)
   
   %v = find(var == i); 
   p = bnet{i}.parents;
   state_idx = bnet{i}.current_state_idx;
   
   if sum(var==i) ~= 1 % only score variables that were not intervened upon        
    if isempty(p) % if has no parents
       prob = prob * bnet{i}.cpd.dists{1}(state_idx);
       
     else % has parents
       % get parent states
       for j = 1 : length(p)
            pstate(j) = bnet{p(j)}.current_state_idx;
       end

       % find row of cpt
       for j = 1 : length(bnet{i}.cpd.parent_settings)
           
           if sum(pstate == bnet{i}.cpd.parent_settings{j}) == length(pstate)
            prob = prob * bnet{i}.cpd.dists{j}(state_idx);
            break;
           end

       end
       
    end
   end
    
end
