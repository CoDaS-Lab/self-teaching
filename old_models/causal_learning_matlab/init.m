function init()

% turn off warnings
warning('off', 'all');

% load libraries  
cd "~/Code/bnt"
addpath(genpathKPM("~/Code/bnt"))
addpath(genpathKPM("~/Code/pmtk3"))

% set plotting device
graphics_toolkit('gnuplot')
setenv("GNUTERM","qt")

% switch back to directory 
cd "~/Dropbox/research/current-projects/self-teaching/original_causal_learning_code/"

end
