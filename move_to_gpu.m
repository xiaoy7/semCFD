function varargout = move_to_gpu(varargin)
varargout = cell(size(varargin));
for k = 1:nargin
    varargout{k} = gpuArray(varargin{k});
end
end

