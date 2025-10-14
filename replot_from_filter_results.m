function stats = replot_from_filter_results(base_output_dir)
%REPLOT_FROM_FILTER_RESULTS Re-create plots using stored filter summaries.
%   REPLOT_FROM_FILTER_RESULTS(BASE_OUTPUT_DIR) scans BASE_OUTPUT_DIR for
%   subdirectories named ``run_###`` that contain ``filter_results.mat``
%   files. The function loads the saved TV and MSE summaries and recreates
%   the per-run and averaged plots without re-running any simulations. The
%   aggregated statistics are saved alongside the figures. When an output
%   argument is requested, a struct mirroring the saved statistics is
%   returned.
%
%   BASE_OUTPUT_DIR defaults to fullfile(pwd, 'multi_run_outputs') when not
%   specified.
%
%   Example
%       replot_from_filter_results;                     % use default path
%       replot_from_filter_results('my_outputs');       % custom directory
%
%   See also RUN_MULTIPLE_SIMULATIONS.

    if nargin < 1 || isempty(base_output_dir)
        base_output_dir = fullfile(pwd, 'multi_run_outputs');
    end

    base_output_dir = char(base_output_dir);

    if ~isfolder(base_output_dir)
        error('Base output directory not found: %s', base_output_dir);
    end

    run_listing = dir(fullfile(base_output_dir, 'run_*'));
    run_listing = run_listing([run_listing.isdir]);

    if isempty(run_listing)
        error('No run directories matching run_* found under %s.', base_output_dir);
    end

    % Sort runs lexicographically to ensure deterministic ordering
    [~, sort_idx] = sort({run_listing.name});
    run_listing = run_listing(sort_idx);

    num_runs = numel(run_listing);
    tv_AB_runs = [];
    tv_A_opt_runs = [];
    tv_B_opt_runs = [];
    mse_A_runs = [];
    mse_B_runs = [];
    obs_indices = [];

    for run_idx = 1:num_runs
        run_dir = fullfile(base_output_dir, run_listing(run_idx).name);
        results_file = fullfile(run_dir, 'filter_results.mat');

        if ~isfile(results_file)
            error('Expected results file not found: %s', results_file);
        end

        data = load(results_file, 'tv_summary', 'mse_summary');

        if ~isfield(data, 'tv_summary')
            error('tv_summary missing from %s.', results_file);
        end
        if ~isfield(data, 'mse_summary')
            error('mse_summary missing from %s.', results_file);
        end

        tv_summary = data.tv_summary;
        mse_summary = data.mse_summary;

        required_tv_fields = {'obs_indices', 'inside_AB', 'inside_A_vs_optimal', 'inside_B_vs_optimal'};
        for k = 1:numel(required_tv_fields)
            if ~isfield(tv_summary, required_tv_fields{k})
                error('tv_summary.%s missing from %s.', required_tv_fields{k}, results_file);
            end
        end

        required_mse_fields = {'methodA', 'methodB'};
        for k = 1:numel(required_mse_fields)
            if ~isfield(mse_summary, required_mse_fields{k})
                error('mse_summary.%s missing from %s.', required_mse_fields{k}, results_file);
            end
        end

        current_indices = tv_summary.obs_indices(:).';

        if isempty(obs_indices)
            obs_indices = current_indices;
            num_obs = numel(obs_indices);
            tv_AB_runs = zeros(num_runs, num_obs);
            tv_A_opt_runs = zeros(num_runs, num_obs);
            tv_B_opt_runs = zeros(num_runs, num_obs);
            mse_A_runs = zeros(num_runs, num_obs);
            mse_B_runs = zeros(num_runs, num_obs);
        else
            if numel(current_indices) ~= numel(obs_indices) || any(current_indices ~= obs_indices)
                error('Observation indices mismatch in %s.', results_file);
            end
        end

        tv_AB_runs(run_idx, :) = tv_summary.inside_AB(:).';
        tv_A_opt_runs(run_idx, :) = tv_summary.inside_A_vs_optimal(:).';
        tv_B_opt_runs(run_idx, :) = tv_summary.inside_B_vs_optimal(:).';

        mse_A_runs(run_idx, :) = mse_summary.methodA(:).';
        mse_B_runs(run_idx, :) = mse_summary.methodB(:).';
    end

    mean_tv_AB = mean(tv_AB_runs, 1);
    mean_tv_A_opt = mean(tv_A_opt_runs, 1);
    mean_tv_B_opt = mean(tv_B_opt_runs, 1);
    mean_mse_A = mean(mse_A_runs, 1);
    mean_mse_B = mean(mse_B_runs, 1);

    colors = lines(num_runs);

    fig_runs_ab = figure('Name', 'Posterior TV distance trajectories (Method A vs B)');
    hold on;
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
        plot(obs_indices, tv_B_opt_runs(run_idx, :), '-', 'Color', colors(run_idx, :) * 0.7, ...
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
            'run_dirs', {fullfile(base_output_dir, {run_listing.name})});
    end
end