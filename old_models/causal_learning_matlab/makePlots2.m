function makePlots()

load newCausalPreds2

% make teaching plots
% compute probabilities for each pair of choices

exnum = [1 5 7];
these{1} = 1:4; these{2} = [5,6,9,10]; these{3} = [7 8 11 12];
causenum = [1 4 7]; % common cause, effect, chain
pairs = [1 1; 1 2; 1 3; 2 2; 2 3; 3 3];
times = [1 2 2 1 2 1];


teach = cell(1,3);
for i = 1 : length(causenum)
    % this assumes independent
    %tmp(i,:) = sum(likelihoods{1}(these{i}, :));
    
    teach{i} = ones(3);
    % this does sequential
    for j = 1 : length(exnum)
        
        for k = 1 : length(exnum)
            teach{i}(j,k) = likelihoods{1}(exnum(j),causenum(i)) .* like2{j}(exnum(k),causenum(i));
        end
    end
    
end


for h = 1 : length(teach)
    props = [];
    for i = 1 : size(teach{h},1)
        for j = i:3
            if i==j
                props(end+1) = teach{h}(i,j);
            else
                props(end+1) = teach{h}(i,j)+teach{h}(j,i);
            end
        end
    end
    
    disp(props);
    bar(props, 'k');
    set(gca, 'xticklabel', {});
    q = axis; q(4) = .5; q(1:2) = [0 7];
    axis(q); box off;
    set(gca, 'ytick', 0:.2:.5,'fontsize', 20);
    print('-depsc', ['new_figure_teaching',num2str(h)]);
end
