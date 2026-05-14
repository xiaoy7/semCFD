function value = read_logical_env(name, defaultValue)
raw = getenv(name);
if isempty(raw)
    value = defaultValue;
    return
end

value = any(strcmpi(raw, {'1', 'true', 'yes', 'on'}));
end

