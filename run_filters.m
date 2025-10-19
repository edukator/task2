function sim_output = run_filters(varargin)
%RUN_FILTERS Run the filtering experiment and record posterior statistics.
%   SIM_OUTPUT = RUN_FILTERS(Name,Value,...) executes one instance of the
%   filtering experiment, saving the signal, observations and posterior
%   summary statistics to disk. In addition to the four barrier methods, an
%   optimal SIR particle filter with a significantly larger ensemble is
%   executed to provide reference posterior mass distributions and
%   normalized filtered-state MSE curves.
%
%   Name/value pairs:
%       'output_dir'     Base directory for result files (default pwd).
%       'results_file'   MAT file used to store summary results (default
%                        'filter_results.mat' inside output_dir).
%       'method_labels'  1x4 cell array with labels for the barrier
%                        filters (default {'MethodA','MethodB','MethodC','MethodD'}).
%
%   Example
%       run_filters;                 % run with defaults
%       run_filters('output_dir', 'tmp/run1');
%
%   The returned struct SIM_OUTPUT contains paths to the generated files
%   together with metadata such as the number of observations.

    parser = inputParser;
    parser.FunctionName = mfilename;

    isStringy = @(s) (ischar(s) || (isstring(s) && isscalar(s)));
    isLabelCell = @(c) iscell(c) && numel(c) == 4 && all(cellfun(isStringy, c));

    addParameter(parser, 'output_dir', pwd, isStringy); % sure directory name is string string
    addParameter(parser, 'results_file', 'filter_results.mat', isStringy);% sure filename is a string
    addParameter(parser, 'method_labels', {'MethodA', 'MethodB', 'MethodC', 'MethodD'}, isLabelCell);

    parse(parser, varargin{:});
    opts = parser.Results;

    output_dir = char(opts.output_dir);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    results_file = char(opts.results_file);
    if isempty(fileparts(results_file))
        results_file = fullfile(output_dir, results_file);
    end
    results_dir = fileparts(results_file);
    if ~isempty(results_dir) && ~isfolder(results_dir)
        mkdir(results_dir);
    end

    method_labels = cellfun(@char, opts.method_labels, 'UniformOutput', false);
    num_methods = 4;
    if numel(method_labels) ~= num_methods
        error('method_labels must contain exactly %d entries.', num_methods);
    end

    %% Parameters
    F = 8;                  % forcing parameter
    he = 1e-3;              % time step, standard Euler
    t_final = 5;           % duration of the simulation in natural time units
    NTe = fix(t_final/he);  % no. of discrete time steps, standard Euler

    sz = sqrt(1/4);         % std of the observations; full filtering
    sx = sqrt(1/2);         % std of the signal noise / diffusion coefficient
    tobs = 0.1;             % continuous time between observations

    % Particle filter parameters
    ness_thr = 0.7;         % NESS threshold for resampling

    % Simulation parameters
    n_steps = ceil(5/he);    % number of time step to skip transient solution
    n_obs = ceil(tobs/he);   % number of subintervals between two observations
    filtered_solution_indices = 1:n_obs:NTe+1; % filtered solution indices
    coarse_time_mesh = he.*(0:n_obs:NTe);      % time mesh for filtered solution

    full_indices = 1:NTe+1;    % ground truth signal and predicted solution indices
    fine_time_mesh = he*(0:NTe);

    %% Barrier parameters
    r_obs = 4*sz;
    barrier_params.p = r_obs;
    barrier_params.alpha = 1;
    barrier_params.mu = 5;
    barrier_params.k = 4;

    optimal_label = 'SIR_Optimal';

    %% Model dimensions and observation pattern
    Dx = 10;
    %N =(2^10);
    N=1000;
    Dz = fix(3*Dx/5);          % number of observed components
    fixed_observed_components = randsample(Dx, Dz);
    fixed_observed_components = sort(fixed_observed_components);

    %% Generate reference trajectory
    ok = 0;
    while ~ok
        n_steps = ceil(5/he);
        Wx0 = sqrt(he)*randn([Dx n_steps]);
        [x_ini, ~] = exp_euler(rand([Dx 1]), he, F, n_steps, Dx, Wx0, sx);
        idx = randsample(fix(n_steps/2):n_steps, 1);
        x0 = x_ini(:, idx);

        Wx = sqrt(he)*randn([Dx NTe]);
        [x, ok] = exp_euler(x0, he, F, NTe, Dx, Wx, sx);
    end

    H0 = eye(Dx) + randn([Dx Dx]).*5e-4;
    H = H0(fixed_observed_components, :);
    Hx = H*x(1:Dx, (n_obs+1):n_obs:NTe+1);
    ze_sparse = Hx + sz*randn(size(Hx));

    %% Initialize particle ensembles
    shared_initial = x0 + sz*randn([Dx N]);

    barrier_params_A = barrier_params;
    barrier_params_B = barrier_params;
    barrier_params_B.p = 0.5 * r_obs;
    barrier_params_C = barrier_params;
    barrier_params_C.p = barrier_params.p / 4;
    barrier_params_D = barrier_params;
    barrier_params_D.p = barrier_params.p / 8;

    barrier_params_list = {barrier_params_A, barrier_params_B, ...
        barrier_params_C, barrier_params_D};

    X0_methods = repmat({shared_initial}, num_methods, 1);

    %% Run filters
    num_obs = size(ze_sparse, 2);
    r = 4;
    r_sub = 2;
    partitions = cell(1, num_obs);
    for obs_idx = 1:num_obs
        cent = x(:, (n_obs) * obs_idx + 1);
        partitions{obs_idx} = hypercube_partition_full(cent, r, r_sub);% holds all info related to geometry of hypercubes
    end

    template_measurement_data = initialize_measurement_storage(num_obs);
    measurement_data_methods = repmat(template_measurement_data, num_methods, 1);
    measurement_data_opt = template_measurement_data;

    record_method = cell(num_methods, 1);
    for method_idx = 1:num_methods
        idx = method_idx;
        record_method{idx} = @(obs_idx, particles, weights) record_measurement(obs_idx, particles, weights, idx); % called by filters, it saves insidemass,insidecount,subcubemass,totalmass
    end
    record_opt = @(obs_idx, particles, weights) record_measurement(obs_idx, particles, weights, num_methods + 1);

    Xf_methods = cell(num_methods, 1);
    for method_idx = 1:num_methods
        idx = method_idx;
        opts_method = struct('measurement_handler', record_method{idx});
        Xf_methods{idx} = sir_barrier(F, sx, sz, he, NTe, n_obs, ze_sparse, H, X0_methods{idx}, ness_thr, barrier_params_list{idx}, opts_method);
    end

    optimal_particle_multiplier = 20;
    N_opt = max(N, ceil(optimal_particle_multiplier * N));
    X0_opt = x0 + sz*randn([Dx N_opt]);
    opts_opt = struct('measurement_handler', record_opt);
    [Xf_opt] = sir(F, sx, sz, he, NTe, n_obs, ze_sparse, H, X0_opt, ness_thr, opts_opt);

    obs_indices = 1:num_obs;
    vs_optimal_distances = zeros(num_methods, num_obs);

    for obs_idx = 1:num_obs
        masses_opt = measurement_data_opt.inside_total_masses{obs_idx};
        method_masses = cell(num_methods, 1);
        for method_idx = 1:num_methods
            method_masses{method_idx} = measurement_data_methods(method_idx).inside_total_masses{obs_idx};
        end

        for method_idx = 1:num_methods
            vs_optimal_distances(method_idx, obs_idx) = 0.5 * sum(abs(method_masses{method_idx} - masses_opt));
        end
    end

    tv_summary = struct('obs_indices', obs_indices, ...
        'method_labels', {method_labels}, ...
        'vs_optimal_distances', vs_optimal_distances, ...
        'optimal_label', optimal_label);

    tv_summary.inside_A_vs_optimal = vs_optimal_distances(1, :);
    tv_summary.inside_B_vs_optimal = vs_optimal_distances(2, :);
    tv_summary.inside_C_vs_optimal = vs_optimal_distances(3, :);
    tv_summary.inside_D_vs_optimal = vs_optimal_distances(4, :);

    Xopt_filtered = Xf_opt(:, 2:end);
    Pd_f = mean(sum(Xopt_filtered.^2, 1));
    if Pd_f <= eps
        Pd_f = 1;
    end

    mse_vs_opt_methods = zeros(num_methods, num_obs);
    for method_idx = 1:num_methods
        mse_vs_opt_methods(method_idx, :) = sum((Xf_methods{method_idx}(:, 2:end) - Xopt_filtered).^2, 1) ./ Pd_f;
    end

    x_truth_coarse = x(1:Dx, (n_obs+1):n_obs:NTe+1);
    PDF_signal = mean(sum(x_truth_coarse.^2, 1));
    if PDF_signal <= eps
        PDF_signal = 1;
    end

    mse_signal_methods = zeros(num_methods, num_obs);
    for method_idx = 1:num_methods
        mse_signal_methods(method_idx, :) = sum((Xf_methods{method_idx}(:, 2:end) - x_truth_coarse).^2, 1) ./ PDF_signal;
    end
    mse_signal_optimal = sum((Xopt_filtered - x_truth_coarse).^2, 1) ./ PDF_signal;

    mse_summary = struct('obs_indices', obs_indices, ...
        'method_labels', {method_labels}, ...
        'optimal_label', optimal_label, ...
        'vs_optimal', mse_vs_opt_methods, ...
        'signal', mse_signal_methods, ...
        'signal_optimal', mse_signal_optimal, ...
        'normalization', Pd_f, ...
        'PDF_signal', PDF_signal);

    mse_summary.methodA = mse_vs_opt_methods(1, :);
    mse_summary.methodB = mse_vs_opt_methods(2, :);
    mse_summary.methodC = mse_vs_opt_methods(3, :);
    mse_summary.methodD = mse_vs_opt_methods(4, :);
    mse_summary.MSE_signal_A = mse_signal_methods(1, :);
    mse_summary.MSE_signal_B = mse_signal_methods(2, :);
    mse_summary.MSE_signal_C = mse_signal_methods(3, :);
    mse_summary.MSE_signal_D = mse_signal_methods(4, :);
    mse_summary.MSE_signal_optimal = mse_signal_optimal;

    method_summaries = repmat(struct('label', '', 'inside_mass', [], ...
        'inside_count', []), num_methods + 1, 1);
    for method_idx = 1:num_methods
        method_summaries(method_idx).label = method_labels{method_idx};
        method_summaries(method_idx).inside_mass = measurement_data_methods(method_idx).inside_mass;
        method_summaries(method_idx).inside_count = measurement_data_methods(method_idx).inside_count;
    end
    method_summaries(num_methods + 1).label = optimal_label;
    method_summaries(num_methods + 1).inside_mass = measurement_data_opt.inside_mass;
    method_summaries(num_methods + 1).inside_count = measurement_data_opt.inside_count;

    %% Save results to MAT file
    results.params = struct( ...
        'F', F, ...
        'he', he, ...
        't_final', t_final, ...
        'NTe', NTe, ...
        'sz', sz, ...
        'sx', sx, ...
        'tobs', tobs, ...
        'ness_thr', ness_thr, ...
        'n_steps', n_steps, ...
        'n_obs', n_obs, ...
        'filtered_solution_indices', filtered_solution_indices, ...
        'coarse_time_mesh', coarse_time_mesh, ...
        'full_indices', full_indices, ...
        'fine_time_mesh', fine_time_mesh, ...
        'barrier_params', barrier_params, ...
        'barrier_params_methods', {barrier_params_list}, ...
        'Dx', Dx, ...
        'N', N, ...
        'Dz', Dz, ...
        'fixed_observed_components', fixed_observed_components, ...
        'method_labels', {method_labels}, ...
        'optimal_label', optimal_label);

    results.truth = struct( ...
        'x_coarse', x_truth_coarse, ...
        'x0', x0);

    results.tv_summary = tv_summary;
    results.method_summaries = method_summaries;
    results.mse_summary = mse_summary;

    save(results_file, '-struct', 'results');

    fprintf('Saved filter results to %s\n', results_file);

    sim_output = struct( ...
        'results_file', results_file, ...
        'method_labels', {method_labels}, ...
        'num_observations', num_obs, ...
        'tv_summary', tv_summary, ...
        'method_summaries', method_summaries, ...
        'mse_summary', mse_summary, ...
        'params', results.params, ...
        'truth', results.truth);

    if nargout == 0
        clear sim_output;
    end

    function data = initialize_measurement_storage(num_obs_local)
        data.inside_mass = zeros(num_obs_local, 1);
        data.inside_count = zeros(num_obs_local, 1);
        data.subcube_masses = cell(num_obs_local, 1);
        data.inside_total_masses = cell(num_obs_local, 1);
    end

    function record_measurement(obs_idx_local, particles_local, weights_local, method_id)
        S_local = partitions{obs_idx_local};
        stats_local = maincube_mass(particles_local, weights_local(:), S_local);
        masses_local = subcube_masses_loop(stats_local.X_in, stats_local.w_in, S_local);
        inside_total_local = stats_local.inside_mass * masses_local;

        if method_id >= 1 && method_id <= num_methods
            measurement_data_methods(method_id).inside_mass(obs_idx_local) = stats_local.inside_mass;
            measurement_data_methods(method_id).inside_count(obs_idx_local) = stats_local.inside_count;
            measurement_data_methods(method_id).subcube_masses{obs_idx_local} = masses_local;
            measurement_data_methods(method_id).inside_total_masses{obs_idx_local} = inside_total_local;
        elseif method_id == num_methods + 1
            measurement_data_opt.inside_mass(obs_idx_local) = stats_local.inside_mass;
            measurement_data_opt.inside_count(obs_idx_local) = stats_local.inside_count;
            measurement_data_opt.subcube_masses{obs_idx_local} = masses_local;
            measurement_data_opt.inside_total_masses{obs_idx_local} = inside_total_local;
        else
            error('Unsupported method identifier: %d', method_id);
        end
    end
end
