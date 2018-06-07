function makePlots()

load causalPreds2

% make teaching plots
% compute probabilities for each pair of choices

exnum = [1 5 7];
these{1} = 1:4; these{2} = [5,6,9,10]; these{3} = [7 8 11 12];
causenum = [1 4 7];
pairs = [1 1; 1 2; 1 3; 2 2; 2 3; 3 3];
times = [1 2 2 1 2 1];

for i = 1 : length(causenum)
    tmp(i,:) = sum(likelihoods{1}(these{i}, :));
end
tmp = normalize(tmp,1);

for h = 1 : length(causenum)
    teach{h} = zeros(3);
    
    for i = 1 : length(exnum)

        for j = 1 : length(exnum)
            teach{h}(i,j) = tmp(i, causenum(h)) .* tmp(j,causenum(h));
        end

    end
    
    for i = 1 : size(pairs,1)
        props(i) = teach{h}(pairs(i,1), pairs(i,2)) .* times(i);
    end
    
    disp(props);
    bar(props, 'k');
    set(gca, 'xticklabel', {});
    q = axis; q(4) = .5; q(1:2) = [0 7];
    axis(q); box off;
    set(gca, 'ytick', 0:.2:.5,'fontsize', 20);
    print('-depsc', ['figure_teaching',num2str(h)]);
end