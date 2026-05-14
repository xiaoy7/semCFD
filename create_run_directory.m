function runDir = create_run_directory(repoDir)
parentDir = fileparts(repoDir);
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');
runDir = fullfile(parentDir, ['time', char(currentTime)]);
if ~exist(runDir, 'dir')
    mkdir(runDir);
end
copyfile(fullfile(repoDir, '*.m'), runDir);
end
