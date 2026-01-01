% ===== portability setup =====
root = projroot();   % uses project root; falls back to script folder
ds = imageDatastore( ...
    fullfile(root,'assets','Cars'), ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg','.png'}, ...
    'LabelSource', 'foldernames');

%Paths
imagePath = fullfile(root,'assets','Cars');
csvPath = fullfile(root, 'assets', 'license.csv');

% Disable Warnings
warning('off', 'all');

% Read in CSV file
data = readtable(csvPath);

% Builds the correct image file path and adds to data table
for i = 1:height(data)
    % Extract filename
    [~, name, ext] = fileparts(data.image_name{i}); 
    % Build full path
    data.image_name{i} = fullfile(root, 'assets', 'Cars', [name, ext]); 
    if ~isfile(data.image_name{i})
        error('File not found: %s', data.image_name{i});
    end
end

% [x, y, width, height]
bboxes = zeros(height(data), 4);  

for i = 1:height(data)
    % Get image metadata
    info = imfinfo(data.image_name{i});
    imgWidth = info.Width;
    imgHeight = info.Height;

    % Convert normalized coordinates to pixel values
    x = data.top_x(i) * imgWidth;
    y = data.top_y(i) * imgHeight;
    box_width = (data.bottom_x(i) - data.top_x(i)) * imgWidth;
    box_height = (data.bottom_y(i) - data.top_y(i)) * imgHeight;

    % Assign new pixel values to boundary box vector
    bboxes(i,:) = [x, y, box_width, box_height];
end

% Add the boundary box column to the data table
data.plate = bboxes;

% Removed uneccessary columns
data = removevars(data, {'Var1','top_x','top_y','bottom_x','bottom_y'});
% Converted image format and bbox for proper YOLO format
data.image_name = cellfun(@char, data.image_name, 'UniformOutput', false);
data.plate = num2cell(data.plate, 2);

% Uses the default random number generator seed
rng("default");
shuffledIndices = randperm(height(data));
idx = floor(0.7 * length(shuffledIndices));

% Makes a training table using 70% of the data
trainingIdx = 1:idx;
trainingDataTbl = data(shuffledIndices(trainingIdx),:);

% Makes a validation table using 15% of the data
validationIdx = idx+1 : idx + floor(0.15 * length(shuffledIndices));
validationDataTbl = data(shuffledIndices(validationIdx),:);

% Makes a testing table using 15% of the data 
testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = data(shuffledIndices(testIdx),:);

% Datastores for image and bbox during model training
imdsTrain = imageDatastore(trainingDataTbl{:,"image_name"});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,"plate"));
imdsValidation = imageDatastore(validationDataTbl{:,"image_name"});
bldsValidation = boxLabelDatastore(validationDataTbl(:,"plate"));
imdsTest = imageDatastore(testDataTbl{:,"image_name"});
bldsTest = boxLabelDatastore(testDataTbl(:,"plate"));

% Combine the datastores
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% Resizes images for training
inputSize = [608 608 3];
className = "plate";

rng("default");

% Resize all training images to inputSize and adjust bboxes
trainingData = transform(trainingData, @(data)preprocessData(data, inputSize));

% Resize all validation images to inputSize and adjust bboxes
validationData = transform(validationData, @(data)preprocessData(data, inputSize));

% Use the preprocessed training data for anchor estimation
trainingDataForEstimation = trainingData;

% Estimate anchors on resized data 
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);


% Split anchors into low res and high res head
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)};

% Created YOLO Object
%detector = yolov4ObjectDetector("tiny-yolov4-coco",className,anchorBoxes,InputSize=inputSize);

% Perform data augmentatoin on the training data
augmentedTrainingData = transform(trainingData,@augmentData);


% Training options for YOLO model
options = trainingOptions("adam", ...
    GradientDecayFactor=0.9, ...
    SquaredGradientDecayFactor=0.999, ...
    InitialLearnRate=0.0006, ...
    LearnRateSchedule="none", ...
    MiniBatchSize=8, ...
    L2Regularization=0.0005, ...
    MaxEpochs=100, ...
    DispatchInBackground=true, ...
    ResetInputNormalization=true, ...
    Shuffle="every-epoch", ...
    VerboseFrequency=20, ...
    ValidationFrequency=100, ...
    CheckpointPath=tempdir, ...
    ValidationData=validationData, ...
    OutputNetwork="best-validation-loss");

% Functions for data augmentation
function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.15]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    % Ensure 3 channels
    if ndims(I) == 2
        I = repmat(I, 1, 1, 3);
    elseif size(I,3) == 1
        I = repmat(I, 1, 1, 3);
    end
    imgSize = size(I);

    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end
end

% Train the YOLO v4 detector
%[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

% Load saved detector
load("plateDetector_v4.mat","detector");

%---------------- Read image ----------------
testImg = imread(fullfile(root, "tests", "gldidydubs651.jpg"));

% Detect plate on image
[b, s, l] = detect(detector, testImg);
D1 = insertObjectAnnotation(testImg, 'rectangle', b, s);

figure;
imshow(D1);
title('Plate Detection');

x = b(1);
y = b(2);
w = b(3);
h = b(4);

% Shrink amounts
shrinkX = 0.24;  
shrinkY = 0.28;   

% Compute new tighter box
x2 = x + w*shrinkX;
y2 = y + h*shrinkY;
w2 = w * (1 - 2*shrinkX);
h2 = h * (1 - 2*shrinkY);

tightBox = [x2, y2, w2, h2];

croppedPlate = imcrop(testImg, tightBox);

figure;
imshow(croppedPlate);
title("Cropped Plate");


% OCR Prep
Img1 = croppedPlate;

if size(Img1, 3) == 3
    Img1 = rgb2gray(Img1);
end

Thresh = graythresh(Img1);
binarizedImage = imbinarize(Img1, Thresh);
grayScaleImage = Img1;
invertedImage = ~binarizedImage;
cleanBinary = bwareaopen(invertedImage, 20);

%///Horizontal Projection
horizontalProjection = sum(binarizedImage, 2);
projectionThresh = 0.05*max(horizontalProjection);
horzRows = horizontalProjection > projectionThresh;
Diff1 = diff([0; horzRows; 0]);
startRows = find(Diff1 == 1);
endRows = find(Diff1 == -1) -1;
[~,index1] = max(endRows-startRows+1);
startsMax1 = startRows(index1);
endsMax1 = endRows(index1);

%///Vertical Projection
verticalProjection = sum(binarizedImage, 1);
projectionThresh = 0.05*max(verticalProjection);
vertRows = (verticalProjection > projectionThresh).';
Diff2 = diff([0; vertRows; 0]);
startRows = find(Diff2 == 1);
endRows = find(Diff2 == -1) -1;
[~,index2] = max(endRows-startRows+1);
startsMax2 = startRows(index2);
endsMax2 = endRows(index2);

% Get image dimensions
binarizedCrop = binarizedImage(startsMax1:endsMax1, startsMax2:endsMax2);
grayCrop = grayScaleImage(startsMax1:endsMax1, startsMax2:endsMax2);

% ADD PADDING - OCR needs breathing room
[imgH, imgW] = size(grayScaleImage);
pad = 15;
startsMax1 = max(1, startsMax1 - pad);
endsMax1 = min(imgH, endsMax1 + pad);
startsMax2 = max(1, startsMax2 - pad);
endsMax2 = min(imgW, endsMax2 + pad);
grayCrop = grayScaleImage(startsMax1:endsMax1, startsMax2:endsMax2);
% Simple contrast enhancement - works better than complex binarization
grayCrop = imadjust(grayCrop, stretchlim(grayCrop, [0.01 0.99]));

% WITH THIS:
Iocr = im2uint8(grayCrop);

[h, w] = size(Iocr);
targetHeight = 60;

aspectRatio = w/h;
targetWidth = round(targetHeight*aspectRatio);

Iocr = imresize(Iocr, [targetHeight, targetWidth]);
Iocr = imsharpen(Iocr, 'Amount', 0.8);
Iocr = medfilt2(Iocr, [3 3]);
% Add white border
Iocr = padarray(Iocr, [15 15], 255, 'both');

figure; 
subplot(1,2,1); imshow(Iocr); title('Grayscale sent to OCR');
subplot(1,2,2); imshow(imbinarize(Iocr)); title('How OCR sees it (binary)');

roi = [1 1 size(Iocr,2) size(Iocr,1)];  % Full image ROI
TextResults = ocr(Iocr, roi, 'CharacterSet','ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789');
plate = regexprep(TextResults.Text,'\s','');
disp(['OCR => "' plate '"']);
disp(['Confidence: ' num2str(mean(TextResults.CharacterConfidences))]);
disp(['Raw text: "' TextResults.Text '"']);
disp(['Character confidences: ' num2str(TextResults.CharacterConfidences')]);
disp(['Words detected: ' num2str(length(TextResults.Words))]);

% If plate came back empty OR low confidence, try inverted:
if strlength(plate)==0 || mean(TextResults.CharacterConfidences) < 0.6
    res2 = ocr(imcomplement(Iocr), roi, ...
        'CharacterSet','ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789');
    plate2 = regexprep(res2.Text, '\s+', '');
    disp("OCR (complement) => " + plate2);
    disp(['Confidence: ' num2str(mean(res2.CharacterConfidences))]);
    disp(['Raw text: "' TextResults.Text '"']);
    disp(['Character confidences: ' num2str(TextResults.CharacterConfidences')]);
    disp(['Words detected: ' num2str(length(TextResults.Words))]);
    
    % Use better result
    if mean(res2.CharacterConfidences) > mean(TextResults.CharacterConfidences)
        plate = plate2;
    end
end

%{
% Average Precision Metrics
detectionResults = detect(detector,testData,Threshold=0.01);
metrics = evaluateObjectDetection(detectionResults,testData);
AP = averagePrecision(metrics);
[precision,recall] = precisionRecall(metrics,ClassName="plate");
figure
plot(recall{:},precision{:})
xlabel("Recall")
ylabel("Precision")
grid on
title(sprintf("Average Precision = %.2f",AP))

% Histogram for OCR
figure;
imhist(testImg);
tOtsu = graythresh(testImg);
hold on;
xline(255*tOtsu, "r", "LineWidth", 2);
title(sprintf("Grayscale Histogram (Otsu = %.3f)", tOtsu));
xlabel("Pixel Intensity");
ylabel("Count");
grid on;
hold off;
%}
