function prob = bnet_score_node(bnet, node_id)

cur_node = bnet{node_id};

%look up parent current state ids
par_state_ids = zeros(1,length(cur_node.parents));

for i=1:length(par_state_ids),
  par = cur_node.parents(i);
  par_state_ids(i) = bnet{par}.current_state_idx;
end

%find the appropriate dist
idx = find_par_state_idx(cur_node, par_state_ids);
dist = cur_node.cpd.dists{idx};

%prob gets the value for the current state idx
prob = dist(cur_node.current_state_idx);


function par_states_idx = find_par_state_idx(node, par_setting)
par_states_idx = -1;
for i=1:length(node.cpd.parent_settings)

  cur_setting = node.cpd.parent_settings{i};
  match = 1;
  for j=1:length(cur_setting),
    if (cur_setting(j) ~= par_setting(j))
      match = 0;
      break;
    end
  end

  if (match == 1)
    par_states_idx = i;
    return;
  end
end
error('No match found', par_setting);
