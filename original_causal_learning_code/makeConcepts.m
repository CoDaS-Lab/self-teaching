function [concepts, bnet] = makeConcepts(b,t)
% fixme: more general parameterization set up? take in baserate &
% transmission??

% common cause
% 1->23
%[p(f|f), p(t|f)] [p(f|t) p(t|t]
tmp = zeros(3);
tmp([1 1],[2 3]) = 1;
concepts(1,:) = tmp(:)';

bnet{1}.node{1}.parents = [];
bnet{1}.node{1}.state_space = {0, 1};
bnet{1}.node{1}.current_state = 0;
bnet{1}.node{1}.current_state_idx = 1;
bnet{1}.node{1}.cpd.parent_settings = {[]};
bnet{1}.node{1}.cpd.dists = {[1-b b]};

bnet{1}.node{2}.parents = [1];
bnet{1}.node{2}.state_space = {0, 1};
bnet{1}.node{2}.current_state = 0;
bnet{1}.node{2}.current_state_idx = 1;
bnet{1}.node{2}.cpd.parent_settings = {[1] [2]};
bnet{1}.node{2}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)]};

bnet{1}.node{3}.parents = [1];
bnet{1}.node{3}.state_space = {0, 1};
bnet{1}.node{3}.current_state = 0;
bnet{1}.node{3}.current_state_idx = 1;
bnet{1}.node{3}.cpd.parent_settings = {[1] [2]};
bnet{1}.node{3}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)]};

% 2->13
tmp = zeros(3);
tmp([2 2],[1 3]) = 1;
concepts(2,:) = tmp(:)';

bnet{2}.node(1:3) = bnet{1}.node([2,1,3]);
bnet{2}.node{1}.parents = [2];
bnet{2}.node{2}.parents = [];
bnet{2}.node{3}.parents = [2];

% 3->12
tmp = zeros(3);
tmp([3 3],[1 2]) = 1;
concepts(3,:) = tmp(:)';

bnet{3}.node(1:3) = bnet{1}.node([3,2,1]);
bnet{3}.node{1}.parents = [3];
bnet{3}.node{2}.parents = [3];
bnet{3}.node{3}.parents = [];

% common effect
% 12->3
tmp = zeros(3);
tmp([1 2],[3 3]) = 1;
concepts(4,:) = tmp(:)';

bnet{4}.node{1}.parents = [];
bnet{4}.node{1}.state_space = {0, 1};
bnet{4}.node{1}.current_state = 0;
bnet{4}.node{1}.current_state_idx = 1;
bnet{4}.node{1}.cpd.parent_settings = {[]};
bnet{4}.node{1}.cpd.dists = {[1-b b]};

bnet{4}.node{2}.parents = [];
bnet{4}.node{2}.state_space = {0, 1};
bnet{4}.node{2}.current_state = 0;
bnet{4}.node{2}.current_state_idx = 1;
bnet{4}.node{2}.cpd.parent_settings = {[]};
bnet{4}.node{2}.cpd.dists = {[1-b b]};

bnet{4}.node{3}.parents = [1 2];
bnet{4}.node{3}.state_space = {0, 1};
bnet{4}.node{3}.current_state = 0;
bnet{4}.node{3}.current_state_idx = 1;
bnet{4}.node{3}.cpd.parent_settings = {[1 1] [1 2] [2 1] [2 2]};
bnet{4}.node{3}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)], ...
    [(1-b)*(1-t) 1-(1-b)*(1-t)], [(1-b)*(1-t)^2 1-(1-b)*(1-t)^2]};

% 13->2
tmp = zeros(3);
tmp([1 3],[2 2]) = 1;
concepts(5,:) = tmp(:)';

bnet{5}.node(1:3) = bnet{4}.node([1,3,2]);
bnet{5}.node{1}.parents = [];
bnet{5}.node{2}.parents = [1 3];
bnet{5}.node{3}.parents = [];

% 23->1
tmp = zeros(3);
tmp([2 3],[1 1]) = 1;
concepts(6,:) = tmp(:)';

bnet{6}.node(1:3) = bnet{4}.node([3,1,2]);
bnet{6}.node{1}.parents = [2 3];
bnet{6}.node{2}.parents = [];
bnet{6}.node{3}.parents = [];

% chain
%1->2->3
tmp = zeros(3);
tmp(1,2)=1; tmp(2,3)=1;
concepts(7,:) = tmp(:)';

bnet{7}.node{1}.parents = [];
bnet{7}.node{1}.state_space = {0, 1};
bnet{7}.node{1}.current_state = 0;
bnet{7}.node{1}.current_state_idx = 1;
bnet{7}.node{1}.cpd.parent_settings = {[]};
bnet{7}.node{1}.cpd.dists = {[1-b b]};

bnet{7}.node{2}.parents = [1];
bnet{7}.node{2}.state_space = {0, 1};
bnet{7}.node{2}.current_state = 0;
bnet{7}.node{2}.current_state_idx = 1;
bnet{7}.node{2}.cpd.parent_settings = {[1] [2]};
bnet{7}.node{2}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)]};

bnet{7}.node{3}.parents = [2];
bnet{7}.node{3}.state_space = {0, 1};
bnet{7}.node{3}.current_state = 0;
bnet{7}.node{3}.current_state_idx = 1;
bnet{7}.node{3}.cpd.parent_settings = {[1] [2]};
bnet{7}.node{3}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)]};

%1->3->2
tmp = zeros(3);
tmp(1,3)=1; tmp(3,2)=1;
concepts(8,:) = tmp(:)';

bnet{8}.node(1:3) = bnet{7}.node([1,3,2]);
bnet{8}.node{1}.parents = [];
bnet{8}.node{2}.parents = [3];
bnet{8}.node{3}.parents = [1];

%2->3->1
tmp = zeros(3);
tmp(2,3)=1; tmp(3,1)=1;
concepts(9,:) = tmp(:)';

bnet{9}.node(1:3) = bnet{7}.node([3,1,2]);
bnet{9}.node{1}.parents = [3];
bnet{9}.node{2}.parents = [];
bnet{9}.node{3}.parents = [2];

%2->1->3
tmp = zeros(3);
tmp(2,1)=1; tmp(1,3)=1;
concepts(10,:) = tmp(:)';

bnet{10}.node(1:3) = bnet{7}.node([2,1,3]);
bnet{10}.node{1}.parents = [2];
bnet{10}.node{2}.parents = [];
bnet{10}.node{3}.parents = [1];

%3->1->2
tmp = zeros(3);
tmp(3,1)=1; tmp(1,2)=1;
concepts(11,:) = tmp(:)';

bnet{11}.node(1:3) = bnet{7}.node([2,3,1]);
bnet{11}.node{1}.parents = [3];
bnet{11}.node{2}.parents = [1];
bnet{11}.node{3}.parents = [];

%3->2->1
tmp = zeros(3);
tmp(3,2) = 1; tmp(2,1)=1;
concepts(12,:) = tmp(:)';

bnet{12}.node(1:3) = bnet{7}.node([3,2,1]);
bnet{12}.node{1}.parents = [2];
bnet{12}.node{2}.parents = [3];
bnet{12}.node{3}.parents = [];

% chain with loop
% 1->2->3, 1->3
tmp = zeros(3);
tmp(1,2)=1; tmp(1,3)=1; tmp(2,3)=1;
concepts(13,:) = tmp(:)';

bnet{13}.node{1}.parents = [];
bnet{13}.node{1}.state_space = {0, 1};
bnet{13}.node{1}.current_state = 0;
bnet{13}.node{1}.current_state_idx = 1;
bnet{13}.node{1}.cpd.parent_settings = {[]};
bnet{13}.node{1}.cpd.dists = {[1-b b]};

bnet{13}.node{2}.parents = [1];
bnet{13}.node{2}.state_space = {0, 1};
bnet{13}.node{2}.current_state = 0;
bnet{13}.node{2}.current_state_idx = 1;
bnet{13}.node{2}.cpd.parent_settings = {[1] [2]};
bnet{13}.node{2}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)]};

bnet{13}.node{3}.parents = [1 2];
bnet{13}.node{3}.state_space = {0, 1};
bnet{13}.node{3}.current_state = 0;
bnet{13}.node{3}.current_state_idx = 1;
bnet{13}.node{3}.cpd.parent_settings = {[1 1], [1 2], [2 1], [2 2]};
bnet{13}.node{3}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)], ...
    [(1-b)*(1-t) 1-(1-b)*(1-t)], [(1-b)*(1-t)^2 1-(1-b)*(1-t)^2]};

%1->3->2, 1->2
tmp = zeros(3);
tmp(1,3)=1; tmp(3,2)=1; tmp(1,2)=1;
concepts(14,:) = tmp(:)';

bnet{14}.node(1:3) = bnet{13}.node([1,3,2]);
bnet{14}.node{1}.parents = [];
bnet{14}.node{2}.parents = [1 3];
bnet{14}.node{3}.parents = [1];

%2->3->1, 2->1
tmp = zeros(3);
tmp(2,3)=1; tmp(3,1)=1; tmp(2,1)=1;
concepts(15,:) = tmp(:)';

bnet{15}.node(1:3) = bnet{13}.node([3,1,2]);
bnet{15}.node{1}.parents = [2 3];
bnet{15}.node{2}.parents = [];
bnet{15}.node{3}.parents = [2];

%2->1->3, 2->3
tmp = zeros(3);
tmp(2,1)=1; tmp(1,3)=1; tmp(2,3)=1;
concepts(116,:) = tmp(:)';

bnet{16}.node(1:3) = bnet{13}.node([2,1,3]);
bnet{16}.node{1}.parents = [2];
bnet{16}.node{2}.parents = [];
bnet{16}.node{3}.parents = [1 2];

%3->1->2, 3->1
tmp = zeros(3);
tmp(3,1)=1; tmp(1,2)=1; tmp(3,2)=1;
concepts(17,:) = tmp(:)';

bnet{17}.node(1:3) = bnet{13}.node([2,3,1]);
bnet{17}.node{1}.parents = [3];
bnet{17}.node{2}.parents = [1 3];
bnet{17}.node{3}.parents = [];

%3->2->1
tmp = zeros(3);
tmp(3,2) = 1; tmp(2,1)=1; tmp(3,1)=1;
concepts(12,:) = tmp(:)';

bnet{18}.node(1:3) = bnet{13}.node([3,2,1]);
bnet{18}.node{1}.parents = [2 3];
bnet{18}.node{2}.parents = [3];
bnet{18}.node{3}.parents = [];

% single cause
% 1->2, 3
%[p(f|f), p(t|f)] [p(f|t) p(t|t]
tmp = zeros(3);
tmp(1,2) = 1;
concepts(19,:) = tmp(:)';

bnet{19}.node{1}.parents = [];
bnet{19}.node{1}.state_space = {0, 1};
bnet{19}.node{1}.current_state = 0;
bnet{19}.node{1}.current_state_idx = 1;
bnet{19}.node{1}.cpd.parent_settings = {[]};
bnet{19}.node{1}.cpd.dists = {[1-b b]};

bnet{19}.node{2}.parents = [1];
bnet{19}.node{2}.state_space = {0, 1};
bnet{19}.node{2}.current_state = 0;
bnet{19}.node{2}.current_state_idx = 1;
bnet{19}.node{2}.cpd.parent_settings = {[1] [2]};
bnet{19}.node{2}.cpd.dists = {[1-b b], [(1-b)*(1-t) 1-(1-b)*(1-t)]};

bnet{19}.node{3}.parents = [];
bnet{19}.node{3}.state_space = {0, 1};
bnet{19}.node{3}.current_state = 0;
bnet{19}.node{3}.current_state_idx = 1;
bnet{19}.node{3}.cpd.parent_settings = {[]};
bnet{19}.node{3}.cpd.dists = {[1-b b]};

% 2->1, 3
tmp = zeros(3);
tmp(2, 1) = 1;
concepts(20,:) = tmp(:)';

bnet{20}.node(1:3) = bnet{19}.node([2,1,3]);
bnet{20}.node{1}.parents = [2];
bnet{20}.node{2}.parents = [];
bnet{20}.node{3}.parents = [];

% 1->3, 2
tmp = zeros(3);
tmp(1,3) = 1;
concepts(21,:) = tmp(:)';

bnet{21}.node(1:3) = bnet{19}.node([1,3,2]);
bnet{21}.node{1}.parents = [];
bnet{21}.node{2}.parents = [];
bnet{21}.node{3}.parents = [1];

% 3->1, 2
tmp = zeros(3);
tmp(3,1) = 1;
concepts(22,:) = tmp(:)';

bnet{22}.node(1:3) = bnet{19}.node([2,3,1]);
bnet{22}.node{1}.parents = [3];
bnet{22}.node{2}.parents = [];
bnet{22}.node{3}.parents = [];

% 3->2, 1
tmp = zeros(3);
tmp(3,2) = 1;
concepts(23,:) = tmp(:)';

bnet{23}.node(1:3) = bnet{19}.node([3,2,1]);
bnet{23}.node{1}.parents = [];
bnet{23}.node{2}.parents = [3];
bnet{23}.node{3}.parents = [];

% 2->3, 1
tmp = zeros(3);
tmp(2,3) = 1;
concepts(24,:) = tmp(:)';

bnet{24}.node(1:3) = bnet{19}.node([3,1,2]);
bnet{24}.node{1}.parents = [];
bnet{24}.node{2}.parents = [];
bnet{24}.node{3}.parents = [2];

% no cause
% 1, 2, 3
tmp = zeros(3);
concepts(25,:) = tmp(:)';

bnet{25}.node{1}.parents = [];
bnet{25}.node{1}.state_space = {0, 1};
bnet{25}.node{1}.current_state = 0;
bnet{25}.node{1}.current_state_idx = 1;
bnet{25}.node{1}.cpd.parent_settings = {[]};
bnet{25}.node{1}.cpd.dists = {[1-b b]};

bnet{25}.node{2}.parents = [];
bnet{25}.node{2}.state_space = {0, 1};
bnet{25}.node{2}.current_state = 0;
bnet{25}.node{2}.current_state_idx = 1;
bnet{25}.node{2}.cpd.parent_settings = {[]};
bnet{25}.node{2}.cpd.dists = {[1-b b]};

bnet{25}.node{3}.parents = [];
bnet{25}.node{3}.state_space = {0, 1};
bnet{25}.node{3}.current_state = 0;
bnet{25}.node{3}.current_state_idx = 1;
bnet{25}.node{3}.cpd.parent_settings = {[]};
bnet{25}.node{3}.cpd.dists = {[1-b b]};