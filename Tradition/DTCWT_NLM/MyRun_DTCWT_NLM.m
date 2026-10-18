%% Batch DT-CWT-NLM denoising for Dataset001/rpwi_ave_4.nii
% Writes the configured outputName below outputRoot and
% never overwrites rpwi_ave_4.nii. NIfTI I/O uses bundled pure MATLAB functions and the
% externally installed onlm MEX binary (Linux x86-64 or Windows x86-64).

clear;
clc;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
datasetRoot = fullfile(scriptDir, '..', '..', 'Data', 'Dataset001');

%% User-adjustable parameters
splits = {'train', 'val'};
inputName = 'rpwi_ave_4.nii';
outputName = 'rpwi_dtcwt_nlm.nii';
outputRoot = fullfile(datasetRoot, 'TraditionResult', 'DTCWT_NLM'); % May be any absolute custom folder
reportName = 'dtcwt_nlm_report.csv';
caseFilter = '';               % Example: 'case_001'; empty means all cases
overwrite = false;
dryRun = false;                % true: estimate sigma only, do not call onlm
maxCases = inf;                % Example: 3 for a short test
maskMode = 'none';             % none / output / input_output; full-FOV denoising by default
maskName = 'm0_foreground_mask.nii'; % Shared foreground mask for optional input/output masking.

searchRadius = 3;              % M: search window side = 2*M+1 (default 7)
smallPatchRadius = 1;          % alpha: small patch side = 3
largePatchRadius = 2;          % large patch side = 5
noiseDivisor = 4;              % onlm level = estimated sigma/noiseDivisor
sigmaMultiplier = 1.0;         % global correction of the automatic estimate
waveletLevels = 3;             % J in the 3-D dual-tree complex wavelet
fusionSlopeNormalized = 0.5;   % raw slope = this value / p99.5 scale

%% Validation and file discovery
if ~any(strcmp(maskMode, {'none', 'output', 'input_output'}))
    error('maskMode must be none, output or input_output.');
end
if ~strcmp(maskMode, 'none')
    [parentOutput, methodOutput] = fileparts(outputRoot);
    outputRoot = fullfile(parentOutput, [methodOutput, '_mask_', maskMode]);
end
if searchRadius < 1 || searchRadius ~= floor(searchRadius)
    error('searchRadius must be a positive integer.');
end
if smallPatchRadius < 0 || smallPatchRadius ~= floor(smallPatchRadius)
    error('smallPatchRadius must be a nonnegative integer.');
end
if largePatchRadius <= smallPatchRadius || largePatchRadius ~= floor(largePatchRadius)
    error('largePatchRadius must be an integer larger than smallPatchRadius.');
end
if noiseDivisor <= 0 || sigmaMultiplier <= 0
    error('noiseDivisor and sigmaMultiplier must be positive.');
end
if waveletLevels < 1 || waveletLevels ~= floor(waveletLevels)
    error('waveletLevels must be a positive integer.');
end
if fusionSlopeNormalized <= 0
    error('fusionSlopeNormalized must be positive.');
end
if ~dryRun && exist('onlm', 'file') ~= 3
    error(['The onlm MEX binary could not be loaded. Confirm that onlm.mexa64 ' ...
           '(Linux) or onlm.mexw64 (Windows) matches this MATLAB platform.']);
end

inputFiles = struct('folder', {}, 'name', {}, 'split', {});
for splitIndex = 1:numel(splits)
    splitName = splits{splitIndex};
    splitDir = fullfile(datasetRoot, splitName);
    if ~isfolder(splitDir)
        error('Dataset split does not exist: %s', splitDir);
    end
    found = dir(fullfile(splitDir, '*', inputName));
    for fileIndex = 1:numel(found)
        caseName = getCaseName(found(fileIndex).folder);
        if isempty(caseFilter) || strcmp(caseName, caseFilter)
            item.folder = found(fileIndex).folder;
            item.name = found(fileIndex).name;
            item.split = splitName;
            inputFiles(end + 1) = item; %#ok<SAGROW>
        end
    end
end

if isempty(inputFiles)
    error('No %s files found under %s.', inputName, datasetRoot);
end
[~, order] = sort(strcat({inputFiles.split}', '/', ...
    cellfun(@getCaseName, {inputFiles.folder}', 'UniformOutput', false)));
inputFiles = inputFiles(order);
inputFiles = inputFiles(1:min(numel(inputFiles), maxCases));

%% Process cases
reportRows = cell(0, 13);
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end
reportPath = fullfile(outputRoot, reportName);
numberOfCases = numel(inputFiles);

for fileIndex = 1:numberOfCases
    item = inputFiles(fileIndex);
    caseName = getCaseName(item.folder);
    inputPath = fullfile(item.folder, item.name);
    % Preserve split/case structure below the selected output root.
    outputPath = fullfile(outputRoot, item.split, caseName, outputName);
    status = '';
    message = '';
    imageSizeText = '';
    scale = NaN;
    sigmaRaw = NaN;
    sigmaNormalized = NaN;
    onlmLevel = NaN;
    numberOfCoefficients = 0;
    fatalException = [];
    stage = 'NIfTI read';

    try
        original = dtcwtNiftiIO('read', inputPath);
        original = squeeze(original);
        if ndims(original) ~= 3
            error('Expected a 3-D volume, got %s.', mat2str(size(original)));
        end
        finiteMask = isfinite(original);
        if ~any(finiteMask(:))
            error('Input contains no finite voxels.');
        end
        original(~finiteMask) = 0;
        foreground = [];
        if ~strcmp(maskMode, 'none')
            stage = 'foreground mask';
            foreground = readForegroundMask(fullfile(item.folder, maskName), inputPath, size(original));
        end

        stage = 'sigma estimation';
        scale = robustPercentileScale(original, finiteMask, 99.5);
        [sigmaRaw, numberOfCoefficients] = estimateSigmaHaar3D(original, finiteMask);
        sigmaRaw = sigmaRaw * sigmaMultiplier;
        sigmaNormalized = sigmaRaw / scale;
        onlmLevel = sigmaRaw / noiseDivisor;
        fusionSlopeRaw = fusionSlopeNormalized / scale;
        imageSizeText = sprintf('%dx%dx%d', size(original, 1), ...
            size(original, 2), size(original, 3));

        fprintf(['[%d/%d] %s/%s: sigmaRaw=%.6g, sigmaNorm=%.6g, ' ...
                 'onlmLevel=%.6g\n'], fileIndex, numberOfCases, item.split, ...
                 caseName, sigmaRaw, sigmaNormalized, onlmLevel);

        if dryRun
            status = 'estimated_only';
        elseif isfile(outputPath) && ~overwrite
            if ~strcmp(maskMode, 'none')
                recordPath = [outputPath, '.mask.mat'];
                if ~isfile(recordPath)
                    error('DTCWT:MaskMismatch', 'Existing prediction has no mask record.');
                end
                record = load(recordPath);
                if ~strcmp(record.maskMode, maskMode) || ~isequal(record.foreground, foreground)
                    error('DTCWT:MaskMismatch', 'Existing prediction uses a different mask.');
                end
            end
            status = 'skipped_exists';
            message = 'Set overwrite=true to replace this output.';
        else
            if strcmp(maskMode, 'input_output')
                original(~foreground) = 0; % Noise and scale remain estimated from the original input.
            end
            % The supplied ONLM MEX expects nonnegative MRI magnitude data.
            % Shift only for ONLM, then undo the shift before DT-CWT fusion.
            offset = max(0, -double(min(original(finiteMask))));
            onlmInput = double(original) + offset;
            stage = 'ONLM';
            smallPatch = onlm(onlmInput, searchRadius, ...
                smallPatchRadius, onlmLevel) - offset;
            largePatch = onlm(onlmInput, searchRadius, ...
                largePatchRadius, onlmLevel) - offset;

            stage = 'DT-CWT';
            denoised = real(DTCWT(double(original), smallPatch, largePatch, ...
                waveletLevels, fusionSlopeRaw));
            if ~isequal(size(denoised), size(original)) || any(~isfinite(denoised(finiteMask)))
                error('DT-CWT-NLM returned a wrong-size or non-finite prediction.');
            end
            denoised(~finiteMask) = 0;
            if ~isempty(foreground)
                denoised(~foreground) = 0;
            end
            stage = 'NIfTI write';
            dtcwtNiftiIO('write', inputPath, single(denoised), outputPath);
            if ~isempty(foreground)
                save([outputPath, '.mask.mat'], 'maskMode', 'maskName', 'foreground');
            end
            status = 'processed';
        end
    catch exception
        status = 'error';
        message = sprintf('[%s] %s: %s', stage, exception.identifier, exception.message);
        fprintf(2, '[%d/%d] ERROR %s/%s: %s\n', fileIndex, numberOfCases, ...
            item.split, caseName, message);
        if fileIndex == 1 || ~any(strcmp(reportRows(:, 12), 'error'))
            fprintf(2, '%s\n', getReport(exception, 'extended', 'hyperlinks', 'off'));
        end
        if strcmp(exception.identifier, 'MATLAB:dispatcher:loadLibrary') || ...
                strcmp(exception.identifier, 'DTCWT:NiftiIO')
            fatalException = exception; % Stop repeated environment failures after saving the report.
        end
    end

    reportRows(end + 1, :) = {item.split, caseName, inputPath, outputPath, ...
        imageSizeText, scale, sigmaRaw, sigmaRaw^2, sigmaNormalized, ...
        onlmLevel, numberOfCoefficients, status, message}; %#ok<SAGROW>
    writeReport(reportPath, reportRows);
    if ~isempty(fatalException)
        rethrow(fatalException);
    end
end

statuses = reportRows(:, 12);
fprintf('Finished: processed=%d, estimated=%d, skipped=%d, errors=%d\n', ...
    sum(strcmp(statuses, 'processed')), sum(strcmp(statuses, 'estimated_only')), ...
    sum(strcmp(statuses, 'skipped_exists')), sum(strcmp(statuses, 'error')));
fprintf('Report: %s\n', reportPath);
fprintf('Quality metrics: run Tradition/evaluate_results.py with the project Python environment.\n');
if any(strcmp(statuses, 'error'))
    error('DTCWT:IncompleteRun', 'Some cases failed; inspect %s before evaluation.', reportPath);
end


function name = getCaseName(folder)
    [~, name] = fileparts(folder);
end


function mask = readForegroundMask(path, sourcePath, expectedShape)
    if ~isfile(path)
        error('DTCWT:MaskMismatch', 'Missing foreground mask: %s', path);
    end
    % dtcwtNiftiIO has already loaded the bundled pure MATLAB NIfTI toolbox.
    info = load_untouch_nii(path);
    ref = load_untouch_nii(sourcePath);
    spatialFields = {'qform_code','sform_code','quatern_b','quatern_c','quatern_d', ...
        'qoffset_x','qoffset_y','qoffset_z','srow_x','srow_y','srow_z'};
    if ~isequal(size(info.img), expectedShape) || ...
            any(abs(double(info.hdr.dime.pixdim(1:4))-double(ref.hdr.dime.pixdim(1:4))) > 1e-4)
        error('DTCWT:MaskMismatch', 'Mask dimensions or voxel sizes differ.');
    end
    for i = 1:numel(spatialFields)
        a = double(info.hdr.hist.(spatialFields{i}));
        b = double(ref.hdr.hist.(spatialFields{i}));
        if any(abs(a(:)-b(:)) > 1e-4)
            error('DTCWT:MaskMismatch', 'Mask spatial header differs: %s', spatialFields{i});
        end
    end
    mask = dtcwtNiftiIO('read', path);
    if any(~isfinite(mask(:))) || any(mask(:) ~= 0 & mask(:) ~= 1) || ~any(mask(:))
        error('DTCWT:MaskMismatch', 'Mask must be nonempty and binary.');
    end
    mask = logical(mask);
end


function scale = robustPercentileScale(volume, finiteMask, percentile)
    values = abs(double(volume(finiteMask)));
    values = values(values > eps('single'));
    if isempty(values)
        error('Cannot determine an intensity scale from an all-zero image.');
    end
    % Linear-interpolation percentile, implemented locally to avoid a
    % Statistics and Machine Learning Toolbox dependency.
    values = sort(values);
    rank = 1 + (numel(values) - 1) * percentile / 100;
    lowerIndex = floor(rank);
    upperIndex = ceil(rank);
    fraction = rank - lowerIndex;
    scale = values(lowerIndex) * (1 - fraction) + values(upperIndex) * fraction;
    if ~isfinite(scale) || scale <= 0
        error('Invalid intensity scale: %.9g.', scale);
    end
end


function [sigma, coefficientCount] = estimateSigmaHaar3D(volume, finiteMask)
    detail = double(volume);
    support = finiteMask & abs(volume) > eps('single');

    for dimension = 1:3
        evenLength = 2 * floor(size(detail, dimension) / 2);
        if evenLength < 2
            error('Dimension %d is too short for Haar estimation.', dimension);
        end
        evenIndex = {':', ':', ':'};
        oddIndex = {':', ':', ':'};
        evenIndex{dimension} = 1:2:evenLength;
        oddIndex{dimension} = 2:2:evenLength;
        detail = (detail(evenIndex{:}) - detail(oddIndex{:})) / sqrt(2);
        support = support(evenIndex{:}) & support(oddIndex{:});
    end

    coefficients = detail(support);
    coefficientCount = numel(coefficients);
    if coefficientCount < 100
        error('Only %d valid Haar coefficients were found.', coefficientCount);
    end
    coefficientMedian = median(coefficients);
    sigma = median(abs(coefficients - coefficientMedian)) / 0.6744897501960817;
    if ~isfinite(sigma) || sigma <= 0
        error('Invalid estimated sigma: %.9g.', sigma);
    end
end


function writeReport(reportPath, rows)
    variableNames = {'split', 'case_name', 'input_path', 'output_path', ...
        'shape', 'scale_p99_5', 'sigma_raw', 'variance_raw', ...
        'sigma_normalized', 'onlm_level', 'haar_coefficients', ...
        'status', 'message'};
    report = cell2table(rows, 'VariableNames', variableNames);
    writetable(report, reportPath);
end
