% find the direction of the accel and flip any non-default direction.
% Baihan Lin
% Columbia University
% October 2017

% Please see attached MATLAB direction.m I created for the flipping. 
% To note that, it requires two other matlab files, loadjson.m and 
% savejson.m to work. I tested on 19 testing sample data and it 
% successfully flipped any non-default direction. 
% All you have to do is to run this once in the accel folder 
% (which I don't have permission), which shouldn't take that long, and 
% continue with all the other analyses we planned. NOTE! Please create a 
% copy of the original folder, because my code find any non-default y 
% direction and REPLACE the y data with the flipped ones.
% 
% More for the usage:
% This is a MATLAB function with four inputs, function direction(thisFolder, 
% timeFrame, threshold, output_flipped_files), all the files needs to 
% include full absolute path, instead of relative path.
% 
% e.g. direction('/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_outbound', 
% 20, 20, '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_output4.txt')
% 
% In this case, timeframe 20 and threshold 20 are the heuristic values 
% I used to get about ~95% flipped (18/19  as right) for the 19 testing samples 
% (unrotated), so you can change accordingly to the percentage of flipped 
% in the real datasets. test_output4.txt would store all the cases that has 
% been flipped, and test_output4.txt.unflipped.txt would store all the cases 
% that has not been changed.
% 
% About how to find the direction:
% 
% I used a time frame to scan through the original accel y-axis, once the sum(y(t:t+frame)) > threshold, I decide this is non-default (while the default I defined is sum(y(t:t+frame)) < -threshold).

function direction(thisFolder, timeFrame, threshold, output_flipped_files)
% e.g. direction('/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_outbound', 20, 20, '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_output4.txt')
%
% test_file = '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_walk_outbound.json';
% test_folder = '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_outbound';
% test_output_files = '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_output4.txt';
% test_timeFrame = 20;
% test_threshold = 20;
% thisFolder = test_folder;
% output_flipped_files = test_output_files;
% timeFrame = test_timeFrame;
% threshold = test_threshold;

% system(['echo >  ' output_flipped_files]);
fprintf('Processing folder %s\n', thisFolder);
filePattern = sprintf('%s/*.tmp', thisFolder);
baseFileNames = dir(filePattern);
numberOfFiles = length(baseFileNames);

for f = 1 : numberOfFiles
    fullFileName = fullfile(thisFolder, baseFileNames(f).name);
    fprintf('Processing file %s\n', fullFileName);
    
    rawData = loadjson(fullFileName,'SimplifyCell',1);
    size = length(rawData);
    time = zeros(1,size);
    acc = zeros(3,size);
    
    for t = 1:size
        time(1,t) = rawData(1,t).timestamp;
        acc(1,t) = rawData(1,t).x;
        acc(2,t) = rawData(1,t).y;
        acc(3,t) = rawData(1,t).z;
    end
    
    for t = 1:size-timeFrame
        if sum(acc(2,t:t+timeFrame-1)) > threshold
            system(['echo ' fullFileName ' >> ' output_flipped_files]);
            
            for t = 1:size
                field1 = 'y';  value1 = -acc(2,t);
                field2 = 'timestamp';  value2 = time(1,t);
                field3 = 'z';  value3 = acc(3,t);
                field4 = 'x';  value4 = acc(1,t);
                flipped_Data(t) = struct(field1,value1,field2,value2,field3,value3,field4,value4);
                
            end
            savejson('',flipped_Data,fullFileName);
            break;
        else if sum(acc(2,t:t+timeFrame-1)) < -threshold
                system(['echo ' fullFileName ' >> ' output_flipped_files '.unflipped.txt']);
                break;
            end
        end
    end
    
end

end


