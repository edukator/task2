function [obs_indices, tv_inside] = compute_posterior_tv_distances(results_file, posterior_root, varargin)
%COMPUTE_POSTERIOR_TV_DISTANCES Total-variation distances along the time series.
%   [OBS_INDICES, TV_INSIDE] = COMPUTE_POSTERIOR_TV_DISTANCES(RESULTS_FILE,
%   POSTERIOR_ROOT) loads the stored posterior ensembles produced by
%   RUN_FILTERS and evaluates the total-variation distance between the two
%   filtering methods at each observation time. When the results file already
%   contains a ``tv_summary`` field (as produced by the in-memory workflow),
%   the precomputed trajectories are returned directly without reading
%   particles from disk. The function returns the observation indices
%   together with the inside-only TV distance (TV_INSIDE), which measures the
%   discrepancy of the probability mass that falls inside the reference
%   hypercube.
%
%   Additional name/value pairs:
%       'method_labels'  - 1x2 cell array with folder labels. Defaults to
%                          {'MethodA','MethodB'}.
%       'radius'         - Radius of the main hypercube (default 6).
%       'subcube_radius' - Radius of the subcubes (default 2).
%
%   The helper is deterministic given the stored particles and can therefore
%   be used repeatedly without re-running the filters.

    parser = inputParser;
    parser.FunctionName = mfilename;

    isStringy = @(s) (ischar(s) || (isstring(s) && isscalar(s)));
    isLabelCell = @(c) iscell(c) && numel(c) == 2 && all(cellfun(isStringy, c));

    addParameter(parser, 'method_labels', {'MethodA', 'MethodB'}, isLabelCell);
    addParameter(parser, 'radius', 6, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'subcube_radius', 2, @(x) isnumeric(x) && isscalar(x) && x > 0);

    parse(parser, varargin{:});
    opts = parser.Results;

    method_labels = cellfun(@char, opts.method_labels, 'UniformOutput', false);

    if nargin < 1 || isempty(results_file)
        error('Results file path must be specified.');
    end

    results_file = char(results_file);

    if ~isfile(results_file)
        error('Results file %s not found.', results_file);
    end

    data = load(results_file);

    if isfield(data, 'tv_summary')
        tv_data = data.tv_summary;
        required_fields = {'obs_indices', 'inside_only'};
        for k = 1:numel(required_fields)
            if ~isfield(tv_data, required_fields{k})
                error('tv_summary in %s is missing the ``%s`` field.', results_file, required_fields{k});
            end
        end

        obs_indices = tv_data.obs_indices(:).';
        tv_inside = tv_data.inside_only(:).';
        return;
    end

    if nargin < 2 || isempty(posterior_root)
        error('Posterior root directory must be specified when ``tv_summary`` is absent.');
    end

    posterior_root = char(posterior_root);

    if ~isfolder(posterior_root)
        error('Posterior directory %s not found.', posterior_root);
    end

    if ~isfield(data, 'truth') || ~isfield(data.truth, 'x_coarse')
        error('Results file %s does not contain the expected ``truth.x_coarse`` field.', results_file);
    end
    x_coarse = data.truth.x_coarse;

    posterior_dir_A = fullfile(posterior_root, method_labels{1});
    posterior_dir_B = fullfile(posterior_root, method_labels{2});

    if ~isfolder(posterior_dir_A)
        error('Posterior directory for %s not found: %s', method_labels{1}, posterior_dir_A);
    end
    if ~isfolder(posterior_dir_B)
        error('Posterior directory for %s not found: %s', method_labels{2}, posterior_dir_B);
    end

    num_obs = size(x_coarse, 2);
    obs_indices = 1:num_obs;
    tv_inside = zeros(1, num_obs);

    for obs_idx = 1:num_obs
        cent = x_coarse(:, obs_idx);
        S = hypercube_partition_full(cent, opts.radius, opts.subcube_radius);

        obs_file_A = fullfile(posterior_dir_A, sprintf('obs_%04d.mat', obs_idx));
        obs_data_A = load(obs_file_A, 'particles', 'weights');
        Xa = obs_data_A.particles;
        wa = obs_data_A.weights;

        obs_file_B = fullfile(posterior_dir_B, sprintf('obs_%04d.mat', obs_idx));
        obs_data_B = load(obs_file_B, 'particles', 'weights');
        Xb = obs_data_B.particles;
        wb = obs_data_B.weights;

        [tv_inside(obs_idx), ~] = tv_distance_on_partition_2step(Xa, wa, Xb, wb, S);
    end
end
