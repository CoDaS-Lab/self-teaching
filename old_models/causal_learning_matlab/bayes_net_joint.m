function joint_score = bayes_net_joint(bayes_net)

% collect up all the CPDs
%
% Note that a markov blanket score function could be
% constructed in a similar fashion - pick out a subset
% of nodes, construct a string which multiplies bnet_score_node
% for just those is in the subset (in this case, from the
% markov blanket).

joint_score = 1;

for i = 1:length(bayes_net),
  joint_score = joint_score * bnet_score_node(bayes_net, i);
end
