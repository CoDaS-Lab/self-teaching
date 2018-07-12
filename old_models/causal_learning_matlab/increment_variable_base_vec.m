function vec = increment_variable_base_vec(vec,base)

% FUNCTION vec = increment_variable_base_vec(vec,base)
%
% takes a fixed length vector and increments the index
% takes variable bases for different slots, as specified
% by vector "base" (note "vec" and "base" must be same length

if length(unique(vec >= base)) > 1
  disp('Error: number exceeds base units!!')
  keyboard
end

% increment
vec(end) = vec(end) + 1;

for i = 1 : length(vec)
  if vec(length(vec)+1-i) >= base(length(vec)+1-i)
    vec(length(vec)+1-i) = 0;
    
    if i < length(vec)
      vec(length(vec)-i) = vec(length(vec)-i) + 1;
    else
      break
    end
    
  else
    break
  end
end
