function p = projroot()
% Return the top-level project folder, even when called from subfolders
try
    prj = matlab.project.currentProject;
    p = prj.RootFolder;
catch
    % Fallback if no project loaded: go two levels up from util/
    p = fileparts(fileparts(mfilename('fullpath')));
end
end
