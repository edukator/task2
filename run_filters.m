function sim_output = run_filters(varargin)
%RUN_FILTERS Run the filtering experiment and record posterior statistics.
%   SIM_OUTPUT = RUN_FILTERS(Name,Value,...) executes one instance of the
%   filtering experiment, saving the signal, observations and posterior
%   summary statistics to disk. Particle ensembles may optionally be stored
%   for each observation, but by default only the information required to
%   plot total-variation distances is kept. In addition to the two barrier
%   methods, an optimal SIR particle filter with a significantly larger
%   ensemble is executed to provide reference posterior mass distributions
%   and normalized filtered-state MSE curves.
%
%   Name/value pairs:
%       'output_dir'       Base directory for result files (default pwd).
%       'posterior_root'   Directory for posterior data (default
%                          fullfile(output_dir,'posterior_data')).
%       'results_file'     MAT file used to store summary results (default
%                          'filter_results.mat' inside output_dir).
%       'method_labels'    1x2 cell array with labels for the two filters
%                          (default {'MethodA','MethodB'}).
%       'overwrite_output' Logical flag indicating whether an existing
%                          posterior directory should be cleared before the
%                          simulation (default true).
%       'save_particle_snapshots'
%                         Logical flag controlling whether individual
%                         posterior ensembles are saved to disk (default
%                         false).
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
    isLabelCell = @(c) iscell(c) && numel(c) == 2 && all(cellfun(isStringy, c));

    addParameter(parser, 'output_dir', pwd, isStringy);
    addParameter(parser, 'posterior_root', '', isStringy);
    addParameter(parser, 'results_file', 'filter_results.mat', isStringy);
    addParameter(parser, 'method_labels', {'MethodA', 'MethodB'}, isLabelCell);
    addParameter(parser, 'overwrite_output', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'save_particle_snapshots', false, @(x) islogical(x) && isscalar(x));

    parse(parser, varargin{:});
    opts = parser.Results;

    output_dir = char(opts.output_dir);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    save_particle_snapshots = logical(opts.save_particle_snapshots);

    if save_particle_snapshots
        if isempty(opts.posterior_root)
            posterior_root = fullfile(output_dir, 'posterior_data');
        else
            posterior_root = char(opts.posterior_root);
        end

        if opts.overwrite_output && isfolder(posterior_root)
            rmdir(posterior_root, 's');
        end
        if ~isfolder(posterior_root)
            mkdir(posterior_root);
        end
    else
        posterior_root = '';
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
    N =1000*(2^10);  
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
    X0A = shared_initial;
    X0B = shared_initial;

    barrier_params_A = barrier_params;
    barrier_params_B = barrier_params;
    barrier_params_B.p = 1.5 * r_obs;

    %% Run filters
    num_obs = size(ze_sparse, 2);
    r = 4;
    r_sub = 2;
    partitions = cell(1, num_obs);
    for obs_idx = 1:num_obs
        cent = x(:, (n_obs) * obs_idx + 1);
        partitions{obs_idx} = hypercube_partition_full(cent, r, r_sub);
    end

    measurement_data_A = initialize_measurement_storage(num_obs);
    measurement_data_B = initialize_measurement_storage(num_obs);
    measurement_data_opt = initialize_measurement_storage(num_obs);

    record_A = @(obs_idx, particles, weights) record_measurement(obs_idx, particles, weights, 1);
    record_B = @(obs_idx, particles, weights) record_measurement(obs_idx, particles, weights, 2);
    record_opt = @(obs_idx, particles, weights) record_measurement(obs_idx, particles, weights, 3);

    optsA = struct('save_to_disk', save_particle_snapshots, 'method_label', method_labels{1}, ...
        'output_dir', posterior_root, 'measurement_handler', record_A, 'store_histories', false);
    [Xf_A] = sir_barrier(F, sx, sz, he, NTe, n_obs, ze_sparse, H, X0A, ness_thr, barrier_params_A, optsA);

    optsB = struct('save_to_disk', save_particle_snapshots, 'method_label', method_labels{2}, ...
        'output_dir', posterior_root, 'measurement_handler', record_B, 'store_histories', false);
    [Xf_B] = sir_barrier(F, sx, sz, he, NTe, n_obs, ze_sparse, H, X0B, ness_thr, barrier_params_B, optsB);

    optimal_particle_multiplier = 4;
    N_opt = max(N, ceil(optimal_particle_multiplier * N));
    X0_opt = x0 + sz*randn([Dx N_opt]);
    opts_opt = struct('save_to_disk', save_particle_snapshots, 'method_label', optimal_label, ...
        'output_dir', posterior_root, 'measurement_handler', record_opt, 'store_histories', false);
    [Xf_opt] = sir(F, sx, sz, he, NTe, n_obs, ze_sparse, H, X0_opt, ness_thr, opts_opt);

    obs_indices = 1:num_obs;
    tv_inside_AB = zeros(1, num_obs);
    tv_inside_A_opt = zeros(1, num_obs);
    tv_inside_B_opt = zeros(1, num_obs);
    for obs_idx = 1:num_obs
        masses_A = measurement_data_A.inside_total_masses{obs_idx};
        masses_B = measurement_data_B.inside_total_masses{obs_idx};
        masses_opt = measurement_data_opt.inside_total_masses{obs_idx};
        tv_inside_AB(obs_idx) = 0.5 * sum(abs(masses_A - masses_B));
        tv_inside_A_opt(obs_idx) = 0.5 * sum(abs(masses_A - masses_opt));
        tv_inside_B_opt(obs_idx) = 0.5 * sum(abs(masses_B - masses_opt));
    end

    tv_summary = struct('obs_indices', obs_indices, ...
        'inside_only', tv_inside_AB, ...
        'inside_AB', tv_inside_AB, ...
        'inside_A_vs_optimal', tv_inside_A_opt, ...
        'inside_B_vs_optimal', tv_inside_B_opt, ...
        'optimal_label', optimal_label);

    Xopt_filtered = Xf_opt(:, 2:end);
    Pd_f = mean(sum(Xopt_filtered.^2, 1));
    if Pd_f <= eps
        Pd_f = 1;
    end
    mse_A = sum((Xf_A(:, 2:end) - Xopt_filtered).^2, 1) ./ Pd_f;
    mse_B = sum((Xf_B(:, 2:end) - Xopt_filtered).^2, 1) ./ Pd_f;
    mse_summary = struct('obs_indices', obs_indices, ...
        'methodA', mse_A, 'methodB', mse_B, 'optimal_label', optimal_label, ...
        'normalization', Pd_f);

    method_summaries = repmat(struct('label', '', 'inside_mass', [], ...
        'inside_count', []), 3, 1);
    method_summaries(1).label = method_labels{1};
    method_summaries(1).inside_mass = measurement_data_A.inside_mass;
    method_summaries(1).inside_count = measurement_data_A.inside_count;
    method_summaries(2).label = method_labels{2};
    method_summaries(2).inside_mass = measurement_data_B.inside_mass;
    method_summaries(2).inside_count = measurement_data_B.inside_count;
    method_summaries(3).label = optimal_label;
    method_summaries(3).inside_mass = measurement_data_opt.inside_mass;
    method_summaries(3).inside_count = measurement_data_opt.inside_count;

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
        'Dx', Dx, ...
        'N', N, ...
        'Dz', Dz, ...
        'fixed_observed_components', fixed_observed_components, ...
        'posterior_root', posterior_root, ...
        'method_labels', {method_labels}, ...
        'optimal_label', optimal_label);

    results.truth = struct( ...
        'x_coarse', x(1:Dx, (n_obs+1):n_obs:NTe+1), ...
        'x0', x0);

    results.tv_summary = tv_summary;
    results.method_summaries = method_summaries;
    results.mse_summary = mse_summary;

    save(results_file, '-struct', 'results');

    fprintf('Saved filter results to %s\n', results_file);

    sim_output = struct( ...
        'results_file', results_file, ...
        'posterior_root', posterior_root, ...
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

        switch method_id
            case 1
                measurement_data_A.inside_mass(obs_idx_local) = stats_local.inside_mass;
                measurement_data_A.inside_count(obs_idx_local) = stats_local.inside_count;
                measurement_data_A.subcube_masses{obs_idx_local} = masses_local;
                measurement_data_A.inside_total_masses{obs_idx_local} = inside_total_local;
            case 2
                measurement_data_B.inside_mass(obs_idx_local) = stats_local.inside_mass;
                measurement_data_B.inside_count(obs_idx_local) = stats_local.inside_count;
                measurement_data_B.subcube_masses{obs_idx_local} = masses_local;
                measurement_data_B.inside_total_masses{obs_idx_local} = inside_total_local;
            case 3
                measurement_data_opt.inside_mass(obs_idx_local) = stats_local.inside_mass;
                measurement_data_opt.inside_count(obs_idx_local) = stats_local.inside_count;
                measurement_data_opt.subcube_masses{obs_idx_local} = masses_local;
                measurement_data_opt.inside_total_masses{obs_idx_local} = inside_total_local;
            otherwise
                error('Unsupported method identifier: %d', method_id);
        end
    end
end
