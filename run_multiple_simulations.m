function stats = run_multiple_simulations(varargin)
%RUN_MULTIPLE_SIMULATIONS Run multiple filter experiments and aggregate metrics.
%   RUN_MULTIPLE_SIMULATIONS(Name,Value,...) executes NUM_RUNS independent
%   simulations of the filtering workflow. Each simulation produces its own
%   output directory before posterior TV distances and normalized filtered
%   MSE curves are computed against the optimal SIR filter. The results are
%   averaged and plotted. Optional name/value pairs:
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
    tv_AB_runs = zeros(num_runs, num_obs);
    tv_A_opt_runs = zeros(num_runs, num_obs);
    tv_B_opt_runs = zeros(num_runs, num_obs);
    mse_A_runs = zeros(num_runs, num_obs);
    mse_B_runs = zeros(num_runs, num_obs);

    for run_idx = 1:num_runs
        current = run_results{run_idx};
        if numel(current.obs_indices) ~= num_obs
            error('Observation count changed between runs (%d vs %d).', ...
                numel(current.obs_indices), num_obs);
        end
        tv_AB_runs(run_idx, :) = current.tv_summary.inside_AB(:).';
        tv_A_opt_runs(run_idx, :) = current.tv_summary.inside_A_vs_optimal(:).';
        tv_B_opt_runs(run_idx, :) = current.tv_summary.inside_B_vs_optimal(:).';
        mse_A_runs(run_idx, :) = current.mse_summary.methodA(:).';
        mse_B_runs(run_idx, :) = current.mse_summary.methodB(:).';
    end

    mean_tv_AB = mean(tv_AB_runs, 1);
    mean_tv_A_opt = mean(tv_A_opt_runs, 1);
    mean_tv_B_opt = mean(tv_B_opt_runs, 1);
    mean_mse_A = mean(mse_A_runs, 1);
    mean_mse_B = mean(mse_B_runs, 1);

    fig_runs_ab = figure('Name', 'Posterior TV distance trajectories (Method A vs B)');
    hold on;
    colors = lines(num_runs);
    for run_idx = 1:num_runs
        plot(obs_indices, tv_AB_runs(run_idx, :), 'Color', colors(run_idx, :), ...
            'LineWidth', 1.0, 'DisplayName', sprintf('Run %d', run_idx));
    end
    hold off;
    xlabel('Observation index');
    ylabel('TV distance (A vs B)');
    title('TV distances between Method A and Method B across simulations');
    legend('Location', 'best');
    grid on;

    fig_runs_opt = figure('Name', 'Posterior TV distance trajectories vs optimal');
    hold on;
    for run_idx = 1:num_runs
        plot(obs_indices, tv_A_opt_runs(run_idx, :), '--', 'Color', colors(run_idx, :), ...
            'LineWidth', 1.0, 'DisplayName', sprintf('Run %d (A vs optimal)', run_idx));
        plot(obs_indices, tv_B_opt_runs(run_idx, :), '-', 'Color', colors(run_idx, :)*0.7, ...
            'LineWidth', 1.0, 'DisplayName', sprintf('Run %d (B vs optimal)', run_idx));
    end
    hold off;
    xlabel('Observation index');
    ylabel('TV distance vs optimal');
    title('TV distances between SIR optimal filter and barrier methods');
    legend('Location', 'bestoutside');
    grid on;

    fig_mean_tv = figure('Name', 'Average posterior TV distances');
    hold on;
    plot(obs_indices, mean_tv_AB, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Method A vs Method B');
    plot(obs_indices, mean_tv_A_opt, '--s', 'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Method A vs optimal');
    plot(obs_indices, mean_tv_B_opt, '-.^', 'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Method B vs optimal');
    hold off;
    xlabel('Observation index');
    ylabel('TV distance');
    title('Average posterior TV distances across simulations');
    legend('Location', 'best');
    grid on;

    fig_mean_mse = figure('Name', 'Average normalized filtered MSE');
    hold on;
    plot(obs_indices, mean_mse_A, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Method A');
    plot(obs_indices, mean_mse_B, '-s', 'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Method B');
    hold off;
    xlabel('Observation index');
    ylabel('Normalized MSE');
    title('Average normalized filtered MSE vs optimal filter');
    legend('Location', 'best');
    grid on;

    stats_file = fullfile(base_output_dir, 'tv_distance_statistics.mat');
    save(stats_file, 'obs_indices', 'tv_AB_runs', 'tv_A_opt_runs', 'tv_B_opt_runs', ...
        'mean_tv_AB', 'mean_tv_A_opt', 'mean_tv_B_opt', 'mse_A_runs', 'mse_B_runs', ...
        'mean_mse_A', 'mean_mse_B');
    fprintf('Saved aggregated statistics to %s\n', stats_file);

    saveas(fig_runs_ab, fullfile(base_output_dir, 'tv_distance_runs_AB.fig'));
    saveas(fig_runs_opt, fullfile(base_output_dir, 'tv_distance_runs_vs_optimal.fig'));
    saveas(fig_mean_tv, fullfile(base_output_dir, 'tv_distance_average.fig'));
    saveas(fig_mean_mse, fullfile(base_output_dir, 'filtered_mse_average.fig'));

    if nargout > 0
        stats = struct(...
            'obs_indices', obs_indices,...
            'tv_AB_runs', tv_AB_runs,...
            'tv_A_opt_runs', tv_A_opt_runs,...
            'tv_B_opt_runs', tv_B_opt_runs,...
            'mse_A_runs', mse_A_runs,...
            'mse_B_runs', mse_B_runs,...
            'mean_tv_AB', mean_tv_AB,...
            'mean_tv_A_opt', mean_tv_A_opt,...
            'mean_tv_B_opt', mean_tv_B_opt,...
            'mean_mse_A', mean_mse_A,...
            'mean_mse_B', mean_mse_B,...
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

    tv_summary = sim_output.tv_summary;
    mse_summary = sim_output.mse_summary;

    run_result = struct('run_idx', run_idx, 'run_dir', run_dir, ...
        'obs_indices', tv_summary.obs_indices, 'tv_summary', tv_summary, ...
        'mse_summary', mse_summary);
end
