function new_net = bayes_net_state_set(bayes_net, vars, vals)

obs_variables = vars;
obs_values = vals;
%set the observed variables to the observed values
for i=1:length(obs_variables)
  bayes_net{obs_variables(i)}.current_state = obs_values(i);
  
  bayes_net{obs_variables(i)}.current_state_idx = ...
      cell_contains(bayes_net{obs_variables(i)}.state_space, obs_values(i));
end

new_net = bayes_net;

end

%NOTE: Not really bool; more like idx
function bool = cell_contains(ccell, val)

bool = 0;
for i=1:length(ccell)
    
    if isequal(ccell{i}, val)
        bool = i;
        return;
    end
end

end
        