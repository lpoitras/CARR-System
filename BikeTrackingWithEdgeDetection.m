function BikeTrackingWithEdgeDetection() %#codegen
% Create system objects used for reading video, loading prerequisite data file, detecting pedestrians, and displaying the results.
videoFile       = '320x240clip3.avi';%'vippedtracking.avi';
%scaleDataFile   = 'pedScaleTable'; % An auxiliary file that helps to determine the size of a pedestrian at different pixel locations.
scaleDataFile   = 'JS3.mat'; % for 320x240 clip
%vid = webcam;
obj = setupSystemObjects(videoFile, scaleDataFile);

% Create an empty array of tracks.
tracks = initializeTracks();

% ID of the next track.
nextId = 1;

% Set the global parameters.  //160, 60, 100, 120
option.ROI = [70,70,115,150];                 % for 320x280 clip
%option.ROI                  = [70,50,100,120];   % A rectangle [x, y, w, h] that limits the processing area to ground locations.
option.scThresh             = 0.3;              % A threshold to control the tolerance of error in estimating the scale of a detected pedestrian.
option.gatingThresh         = 0.9;              % A threshold to reject a candidate match between a detection and a track.
option.gatingCost           = 100;              % A large value for the assignment cost matrix that enforces the rejection of a candidate match.
option.costOfNonAssignment  = 10;               % A tuning parameter to control the likelihood of creation of a new track.
option.timeWindowSize       = 16;               % A tuning parameter to specify the number of frames required to stabilize the confidence score of a track.
option.confidenceThresh     = 2;                % A threshold to determine if a track is true positive or false alarm.
option.ageThresh            = 8;                % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.6;              % A threshold to determine the minimum visibility value for a track being true positive.

frameNo = 0;
% Detect people and track them across video frames.
cont = ~isDone(obj.reader);
while cont
    frame   = readFrame();
    frameNo = frameNo+1;
    
    [centroids, bboxes, scores] = detectPeople();
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    
    displayTrackingResults();
    
    % Exit the loop if the video player figure is closed by user.
    cont = ~isDone(obj.reader);% && isOpen(obj.videoPlayer);
end
    function obj = setupSystemObjects(videoFile, scaleDataFile)
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the
        % detected and tracked people in each frame, and playing the video.
        
        % Create a video file reader.
        obj.reader = vision.VideoFileReader(videoFile, 'VideoOutputDataType', 'uint8');
        
        % Create a video player.
        obj.videoPlayer = vision.VideoPlayer('Position', [200, 200, 400, 250]);
        
        % Create a people detector and initialize the parameters.
        % The height of a person in the video is expected to be between [96, 384] pixels.
        obj.peopleDetector = vision.PeopleDetector('UprightPeople_96x48',...
            'MinSize',[96,48],'MaxSize',[384,192], ...
            'ClassificationThreshold',1.1, ...
            'MergeDetections',false);
        
        % Load the scale data file
        ld = load(scaleDataFile, 'JS3');
        obj.JS3 = ld.JS3;
    end
    function tracks = initializeTracks()
        % Create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'color', {}, ...
            'bboxes', {}, ...
            'scores', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'confidence', {}, ...
            'predPosition', {});
    end
    function frame = readFrame()
        frame = step(obj.reader);
    end
    function [centroids, bboxes, scores] = detectPeople()
        % Only extract the region of interest to speed up processing.
        frameToDet = frame(option.ROI(2):option.ROI(2)+option.ROI(4), ...
            option.ROI(1):option.ROI(1)+option.ROI(3), :);
        
        resizeRatio = 4;
        
        % Resize the image to increase the resolution of the pedestrian.
        % vision.PeopleDetector requires minimum size to operate.
        frameToDet = imresize(frameToDet, resizeRatio);
        
        % Run the people detector to produce detection candidates.
        [bboxes, scores] = step(obj.peopleDetector, frameToDet);
        
        % Resize back the output bounding boxes
        bboxes = bboxes / resizeRatio;
        
        % Shift the position of the bounding box back to the original frame.
        bboxes(:,1) = bboxes(:,1) + option.ROI(1);
        bboxes(:,2) = bboxes(:,2) + option.ROI(2);
        
        % Look up the estimated height of a pedestrian based on location of their feet.
        height = bboxes(:, 4);
        yfoot = min(length(obj.JS3), round(bboxes(:,2)+height));
        estHeight = obj.JS3(yfoot);
        
        % Remove detections whose size deviates from the expected size,
        % provided by the calibrated scale estimation.
        invalid = abs(estHeight-height)>estHeight*option.scThresh;
        bboxes(invalid, :) = [];
        scores(invalid, :) = [];
        
        % Apply non-maximum suppression to select the strongest bounding boxes.
        [bboxes, scores] = selectStrongestBbox(bboxes, scores, ...
            'RatioType', 'Min', 'OverlapThreshold', 0.7);
        
        % Compute the centroids
        if isempty(bboxes)
            centroids = [];
        else
            centroids = [(bboxes(:, 1) + bboxes(:, 3) / 2), ...
                (bboxes(:, 2) + bboxes(:, 4) / 2)];
        end
    end
    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            % Get the last bounding box on this track.
            bbox = tracks(i).bboxes(end, :);
            
            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            % Shift the bounding box so that its center is at the predicted
            % location.
            tracks(i).predPosition = [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
        end
    end
    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        % Compute the overlap ratio between the predicted boxes and the
        % detected boxes, and compute the cost of assigning each detection
        % to each track. The cost is minimum when the predicted bbox is
        % perfectly aligned with the detected bbox (overlap ratio is one)
        predBboxes = reshape([tracks(:).predPosition], 4, [])';
        cost = 1 - bboxOverlapRatio(predBboxes, bboxes);
        
        % Force the optimization step to ignore some matches by
        % setting the associated cost to be a large number. Note that this
        % number is different from the 'costOfNonAssignment' below.
        % This is useful when gating (removing unrealistic matches)
        % technique is applied.
        cost(cost > option.gatingThresh) = 1 + option.gatingCost;
        
        % Solve the assignment problem.
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, option.costOfNonAssignment);
    end
    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Stabilize the bounding box by taking the average of the size
            % of recent (up to) 4 boxes on the track.
            T = min(size(tracks(trackIdx).bboxes,1), 4);
            w = mean([tracks(trackIdx).bboxes(end-T+1:end, 3); bbox(3)]);
            h = mean([tracks(trackIdx).bboxes(end-T+1:end, 4); bbox(4)]);
            tracks(trackIdx).bboxes(end+1, :) = [centroid - [w, h]/2, w, h];
            
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            
            % Adjust track confidence score based on the maximum detection
            % score in the past 'timeWindowSize' frames.
            T = min(option.timeWindowSize, length(tracks(trackIdx).scores));
            score = tracks(trackIdx).scores(end-T+1:end);
            tracks(trackIdx).confidence = [max(score), mean(score)];
        end
    end
    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            tracks(idx).bboxes = [tracks(idx).bboxes; tracks(idx).predPosition];
            tracks(idx).scores = [tracks(idx).scores; 0];
            
            % Adjust track confidence score based on the maximum detection
            % score in the past 'timeWindowSize' frames
            T = min(option.timeWindowSize, length(tracks(idx).scores));
            score = tracks(idx).scores(end-T+1:end);
            tracks(idx).confidence = [max(score), mean(score)];
        end
    end
    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age]';
        totalVisibleCounts = [tracks(:).totalVisibleCount]';
        visibility = totalVisibleCounts ./ ages;
        
        % Check the maximum detection confidence score.
        confidence = reshape([tracks(:).confidence], 2, [])';
        maxConfidence = confidence(:, 1);
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages <= option.ageThresh & visibility <= option.visThresh) | ...
            (maxConfidence <= option.confidenceThresh);
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end
    function createNewTracks()
        unassignedCentroids = centroids(unassignedDetections, :);
        unassignedBboxes = bboxes(unassignedDetections, :);
        unassignedScores = scores(unassignedDetections);
        
        for i = 1:size(unassignedBboxes, 1)
            centroid = unassignedCentroids(i,:);
            bbox = unassignedBboxes(i, :);
            score = unassignedScores(i);
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [2, 1], [5, 5], 100);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'color', 255*rand(1,3), ...
                'bboxes', bbox, ...
                'scores', score, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'confidence', [score, score], ...
                'predPosition', bbox);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack; %#ok<AGROW>
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end
    function displayTrackingResults()
        
            if ~isempty(tracks),
                ages = [tracks(:).age]';
                confidence = reshape([tracks(:).confidence], 2, [])';
                maxConfidence = confidence(:, 1);
                avgConfidence = confidence(:, 2);
                opacity = min(0.5,max(0.1,avgConfidence/3));
                noDispInds = (ages < option.ageThresh & maxConfidence < option.confidenceThresh) | ...
                    (ages < option.ageThresh / 2);
                
               
                for i = 1:length(tracks)
                    if ~noDispInds(i)
                       
                 
                        disp(frameNo);
                        
                         
%                         frame = insertShape(frame, ...
%                             'rectangle', tracks(i).bboxes(end, :), ...
%                             'Color', tracks(i).color, ...
%                             'Opacity', opacity(i));
%                         frame = insertObjectAnnotation(frame, ...
%                             'rectangle', tracks(i).bboxes(end, :), ...
%                             num2str(avgConfidence(i)), ...
%                             'Color', tracks(i).color);
                    end
                end
            end
           
            
            % frame = insertText(frame,[10, 10],num2str(frameNo));
            frame = insertShape(frame, 'Rectangle', [option.ROI(1:2), option.ROI(3:4)], ...
                'Color', [255, 0, 0], 'LineWidth', 2);
         step(obj.videoPlayer, frame);
       
    end
end