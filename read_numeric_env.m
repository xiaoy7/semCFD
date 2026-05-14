function value = read_numeric_env(name, defaultValue)
raw = getenv(name);
if isempty(raw)
    value = defaultValue;
    return
end

parsed = str2double(raw);
if isnan(parsed)
    warning('Ignoring invalid numeric environment variable %s=%s', name, raw);
    value = defaultValue;
else
    value = parsed;
end
end

