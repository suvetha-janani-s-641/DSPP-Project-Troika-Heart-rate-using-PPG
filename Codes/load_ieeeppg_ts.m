function [ieeeppg_data, bpm_true] = load_ieeeppg_ts(ts_filename)
% Parses Monash TSER IEEEPPG .ts file and caches to .mat for fast subsequent runs
    cache_file = [ts_filename, '_cache.mat'];
    if exist(cache_file, 'file')
        cached = load(cache_file, 'ieeeppg_data', 'bpm_true');
        ieeeppg_data = cached.ieeeppg_data;
        bpm_true     = cached.bpm_true;
        return;
    end

    raw_text = fileread(ts_filename);
    data_pos = strfind(lower(raw_text), '@data');
    if isempty(data_pos)
        error('No @data tag found in %s', ts_filename);
    end

    body  = strtrim(raw_text(data_pos(1) + 5 : end));
    lines = splitlines(body);
    lines = lines(~cellfun(@isempty, strtrim(lines)));

    num_windows = length(lines);
    first_parts = strsplit(strtrim(lines{1}), ':');
    win_len     = numel(sscanf(first_parts{1}, '%f,'));

    ieeeppg_data = zeros(num_windows, win_len, 5);
    bpm_true     = nan(num_windows, 1);

    for w = 1:num_windows
        parts = strsplit(strtrim(lines{w}), ':');
        for ch = 1:5
            vals = sscanf(parts{ch}, '%f,');
            ieeeppg_data(w, 1:numel(vals), ch) = vals';
        end
        if numel(parts) >= 6 && ~isempty(parts{6})
            bpm_true(w) = str2double(parts{6});
        end
    end

    save(cache_file, 'ieeeppg_data', 'bpm_true');
end