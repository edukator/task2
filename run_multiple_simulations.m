function stats = run_multiple_simulations(varargin)
%RUN_MULTIPLE_SIMULATIONS Run multiple filter experiments and aggregate TV distances.
%   RUN_MULTIPLE_SIMULATIONS(Name,Value,...) executes NUM_RUNS independent
%   simulations of the filtering workflow. Each simulation produces its own
%   output directory before the posterior TV distances are computed. The
%   results are averaged and plotted. Optional name/value pairs:
%
%       'num_runs'        Number of simulations to execute (default 5).
%       'parallel_mode'   Execution mode: 'serial', 'parfor' or 'auto'
%                         (default). 'auto' uses PARFOR when the Parallel
%                         Computing Toolbox is available and NUM_RUNS>1.
%       'base_output_dir' Directory under which per-run outputs are stored
%                         (default fullfile(pwd,'multi_run_outputs')).
%
%   The function returns a struct with aggregated statistics when called
%   with an output argument.
    tStart = tic;
    parser = inputParser;
    parser.FunctionName = mfilename;

    addParameter(parser, 'num_runs', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x, 1) == 0);
    addParameter(parser, 'parallel_mode', 'auto', @(s) (ischar(s) || (isstring(s) && isscalar(s))));
    addParameter(parser, 'base_output_dir', fullfile(pwd, 'multi_run_outputs'), ...
        @(s) (ischar(s) || (isstring(s) && isscalar(s))));

    parse(parser, varargin{:});
    opts = parser.Results;

    num_runs = double(opts.num_runs);
    parallel_mode = lower(char(opts.parallel_mode));
    base_output_dir = char(opts.base_output_dir);

    if ~isfolder(base_output_dir)
        mkdir(base_output_dir);
    end

    run_dirs = arrayfun(@(idx) fullfile(base_output_dir, sprintf('run_%03d', idx)), ...
        1:num_runs, 'UniformOutput', false);

    has_parallel_toolbox = ~isempty(ver('parallel'));
    switch parallel_mode
        case 'serial'
            use_parallel = false;
        case 'parfor'
            if ~has_parallel_toolbox
                error(['Parallel mode requested but the Parallel Computing Toolbox is not ', ...
                    'available.']);
            end
            if num_runs == 1
                warning('Parallel mode requested with a single run; falling back to serial execution.');
                use_parallel = false;
            else
                use_parallel = true;
            end
        case 'auto'
            use_parallel = has_parallel_toolbox && num_runs > 1;
        otherwise
            error('Unsupported parallel_mode value: %s', parallel_mode);
    end

    if use_parallel
        pool = gcp('nocreate');
        if isempty(pool)
            parpool;
        end
    end

    run_results = cell(num_runs, 1);

    if use_parallel
        parfor run_idx = 1:num_runs 
            run_results{run_idx} = execute_single_run(run_idx, run_dirs{run_idx});
        end
    else
        for run_idx = 1:num_runs
            fprintf('Running simulation %d/%d...\n', run_idx, num_runs);
            run_results{run_idx} = execute_single_run(run_idx, run_dirs{run_idx});
        end
    end

    obs_indices = run_results{1}.obs_indices;
    num_obs = numel(obs_indices);
    tv_inside_runs = zeros(num_runs, num_obs);

    for run_idx = 1:num_runs
        current = run_results{run_idx};
        if numel(current.obs_indices) ~= num_obs
            error('Observation count changed between runs (%d vs %d).', ...
                numel(current.obs_indices), num_obs);
        end
        tv_inside_runs(run_idx, :) = current.tv_inside(:).';
    end

    mean_inside = mean(tv_inside_runs, 1);

    fig_runs = figure('Name', 'Posterior TV distance trajectories (inside-only)');
    hold on;
    colors = lines(num_runs);
    for run_idx = 1:num_runs
        plot(obs_indices, tv_inside_runs(run_idx, :), 'Color', colors(run_idx, :), ...
            'LineWidth', 1.0, 'DisplayName', sprintf('Run %d', run_idx));
    end
    hold off;
    xlabel('Observation index');
    ylabel('TV distance (inside-only)');
    title('Inside-only TV distance trajectories across simulations');
    legend('Location', 'best');
    grid on;

    fig_mean = figure('Name', 'Average posterior TV distances (inside-only)');
    hold on;
    plot(obs_indices, mean_inside, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Inside-only mean');
    hold off;
    xlabel('Observation index');
    ylabel('TV distance (inside-only)');
    title('Average inside-only TV distances across simulations');
    legend('Location', 'best');
    grid on;

    stats_file = fullfile(base_output_dir, 'tv_distance_statistics.mat');
    save(stats_file, 'obs_indices', 'tv_inside_runs', 'mean_inside');
    fprintf('Saved aggregated TV distance statistics to %s\n', stats_file);

    saveas(fig_runs, fullfile(base_output_dir, 'tv_distance_runs.fig'));
    saveas(fig_mean, fullfile(base_output_dir, 'tv_distance_average.fig'));

    if nargout > 0
        stats = struct(...
            'obs_indices', obs_indices,...
            'tv_inside_runs', tv_inside_runs,...
            'mean_inside', mean_inside,...
            'base_output_dir', base_output_dir,...
            'run_dirs', {run_dirs});
    end
    elapsed = toc(tStart);
    fprintf('Wall-clock time: %.3f s\n', elapsed);
end

function run_result = execute_single_run(run_idx, run_dir)
%EXECUTE_SINGLE_RUN Helper that runs one simulation and computes TV distances.

    if ~isfolder(run_dir)
        mkdir(run_dir);
    end

    results_file = fullfile(run_dir, 'filter_results.mat');
    posterior_root = fullfile(run_dir, 'posterior_data');

    sim_output = run_filters('output_dir', run_dir, 'results_file', results_file, ...
        'posterior_root', posterior_root, 'overwrite_output', true);

    [obs_indices, tv_inside] = compute_posterior_tv_distances( ...
        sim_output.results_file, sim_output.posterior_root, ...
        'method_labels', sim_output.method_labels);

    run_result = struct('run_idx', run_idx, 'run_dir', run_dir, ...
        'obs_indices', obs_indices, 'tv_inside', tv_inside);
end