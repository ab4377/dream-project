% find the direction of the accel and flip any non-default direction.
% Baihan Lin
% Columbia University
% October 2017

% UPDATE: this matlab file no longer require loadjson and savejson, but it
% requires csvwrite_with_headers.m
%
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
% I used a time frame to scan through the original accel y-axis, once the 
% sum(y(t:t+frame)) > threshold, I decide this is non-default (while the 
% default I defined is sum(y(t:t+frame)) < -threshold).

function direction(thisFolder, timeFrame, threshold, output_flipped_files)
% e.g. direction('/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_outbound', 20, 20, '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_output4.txt')
% direction('/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_csv', 20, 20, '/Users/DoerLBH/Dropbox/git/DREAM_PDBiomarker/test_output6.txt')

topLevelFolder = thisFolder;
[~,list] = system(['find ' topLevelFolder ' -type f -name "*.csv"']);
baseFileNames = strsplit(list);
baseFileNames = baseFileNames(~cellfun('isempty',baseFileNames));
numberOfcsvFiles = length(baseFileNames);

fprintf('Processing folder %s\n', thisFolder);

if numberOfcsvFiles >= 1
    parfor f = 1 : numberOfcsvFiles
        fullFileName = fullfile(char(baseFileNames(f)));
        fprintf('Processing file %s\n', fullFileName);

        data = csvread(fullFileName,1,0);
        fullSize = size(data,1);
        
        seq = data(:,1);
        time = data(:,2);
        acc = data(:,3:5);
        
        for t = 1:fullSize - timeFrame
            if sum(acc(t:t+timeFrame-1,2)) > threshold
                system(['echo ' fullFileName ' >> ' output_flipped_files]);
                flipped_data = [seq, time, acc(:,1), -acc(:,2),acc(:,3)];
                
                headers = {'','timestamp','x','y','z'};
                csvwrite_with_headers(fullFileName, flipped_data, headers);
                break;
            else if sum(acc(t:t+timeFrame-1,2)) < -threshold
                    system(['echo ' fullFileName ' >> ' output_flipped_files '.unflipped.txt']);
                    break;
                end
            end
        end
        
    end
end

end


