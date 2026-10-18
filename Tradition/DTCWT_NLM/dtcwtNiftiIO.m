function result = dtcwtNiftiIO(action, sourcePath, volume, outputPath)
% Pure MATLAB NIfTI-1 I/O; avoids R2018b niftiinfo/affine3d loader failures.
% Uses bundled Jimmy Shen untouch I/O without resampling or reorientation.
% Apply scl_slope/scl_inter once on read; save physical values with slope=1.
    vendorDir = fullfile(fileparts(mfilename('fullpath')), 'vendor', 'NIfTI_20140122');
    if ~strcmp(fileparts(which('load_untouch_nii')), vendorDir) || ...
            ~strcmp(fileparts(which('save_untouch_nii')), vendorDir)
        addpath(vendorDir, '-begin');
    end
    if ~any(strcmp(action, {'read', 'write'}))
        error('DTCWT:NiftiIO', 'Unknown action: %s', action);
    end
    nii = load_untouch_nii(sourcePath);
    dim = double(nii.hdr.dime.dim);
    if nii.filetype ~= 2 || ~any(dim(1) == [3, 4]) || ...
            (dim(1) == 4 && dim(5) ~= 1) || any(dim(2:4) < 1)
        error('DTCWT:NiftiIO', 'Expected single-file NIfTI-1, 3D or singleton-time 4D.');
    end
    if ~any(double(nii.hdr.dime.datatype) == [2,4,8,16,64,256,512,768,1024,1280])
        error('DTCWT:NiftiIO', 'Only real scalar NIfTI data are supported.');
    end
    expectedShape = dim(2:4);
    if strcmp(action, 'read')
        raw = double(nii.img);
        slope = double(nii.hdr.dime.scl_slope);
        intercept = double(nii.hdr.dime.scl_inter);
        if isinf(slope)
            error('DTCWT:NiftiIO', 'Invalid infinite NIfTI scaling slope.');
        elseif isfinite(slope) && slope ~= 0
            if ~isfinite(intercept)
                error('DTCWT:NiftiIO', 'Invalid NIfTI scaling intercept.');
            end
            raw = raw * slope + intercept;
        end % slope=0 or NaN means no scaling; intercept is ignored.
        result = reshape(single(raw), expectedShape);
        if any(isfinite(raw(:)) & ~isfinite(result(:)))
            error('DTCWT:NiftiIO', 'Input signal overflows float32 after scaling.');
        end
        return;
    end
    if ~isnumeric(volume) || ~isreal(volume) || ndims(volume) > 3 || ...
            ~isequal([size(volume,1),size(volume,2),size(volume,3)], expectedShape)
        error('DTCWT:NiftiIO', 'Prediction shape/type differs from the source.');
    end
    volume = single(volume);
    if any(~isfinite(volume(:)))
        error('DTCWT:NiftiIO', 'Refusing to save NaN/Inf or float32 overflow.');
    end
    [outFolder, ~, extension] = fileparts(outputPath);
    if ~strcmpi(extension, '.nii')
        error('DTCWT:NiftiIO', 'Dataset001 predictions must use the .nii extension.');
    end
    if isempty(outFolder), outFolder = pwd; end
    if ~isfolder(outFolder), mkdir(outFolder); end
    % Canonical paths protect the source even with relative paths/symlinks.
    sourceFile = java.io.File(sourcePath);
    outputFile = java.io.File(outputPath);
    if strcmp(char(sourceFile.getCanonicalPath()), char(outputFile.getCanonicalPath()))
        error('DTCWT:NiftiIO', 'Refusing to overwrite the source NIfTI.');
    end
    nii.img = volume;
    nii.hdr.dime.datatype = 16;
    nii.hdr.dime.bitpix = 32;
    nii.hdr.dime.scl_slope = 1;
    nii.hdr.dime.scl_inter = 0;
    % Preserve dimensions (including singleton time), voxel sizes, qform/sform
    % and extensions. Header orientation is never applied to the data array.
    temporary = [tempname(outFolder), '.nii'];
    cleanup = onCleanup(@() removeTemporary(temporary)); %#ok<NASGU>
    save_untouch_nii(nii, temporary);
    [ok, message] = movefile(temporary, outputPath, 'f');
    if ~ok, error('DTCWT:NiftiIO', 'Cannot move prediction into place: %s', message); end
    result = [];
end

function removeTemporary(path)
    if isfile(path), delete(path); end
end
