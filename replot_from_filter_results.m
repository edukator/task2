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
    obs_indices = [];

    % Storage for the legacy two-method layout
    tv_AB_runs = [];
    tv_A_opt_runs = [];
    tv_B_opt_runs = [];
    mse_A_runs = [];
    mse_B_runs = [];
    mse_signal_A_runs = [];
    mse_signal_B_runs = [];

    % Storage for the extended multi-method layout
    tv_pair_runs = [];
    tv_vs_opt_runs = [];
    pairwise_labels = {};
    method_labels = {};
    optimal_label = '';
    mse_vs_opt_runs = [];
    mse_signal_runs = [];
    mse_signal_opt_runs = [];

    using_new_tv = [];
    using_new_mse = [];

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

        if ~isfield(tv_summary, 'obs_indices')
            error('tv_summary.obs_indices missing from %s.', results_file);
        end
        current_indices = tv_summary.obs_indices(:).';

        if isempty(obs_indices)
            obs_indices = current_indices;
            num_obs = numel(obs_indices);
        else
            if numel(current_indices) ~= numel(obs_indices) || any(current_indices ~= obs_indices)
                error('Observation indices mismatch in %s.', results_file);
            end
        end

        current_new_tv = isfield(tv_summary, 'pairwise_distances') && ...
            isfield(tv_summary, 'method_labels') && ...
            isfield(tv_summary, 'vs_optimal_distances');
        if isempty(using_new_tv)
            using_new_tv = current_new_tv;
        elseif using_new_tv ~= current_new_tv
            error('Mixed tv_summary formats encountered; rerun simulations for consistency.');
        end

        current_new_mse = isfield(mse_summary, 'vs_optimal') && ...
            isfield(mse_summary, 'signal') && ...
            isfield(mse_summary, 'signal_optimal');
        if isempty(using_new_mse)
            using_new_mse = current_new_mse;
        elseif using_new_mse ~= current_new_mse
            error('Mixed mse_summary formats encountered; rerun simulations for consistency.');
        end

        if using_new_tv
            current_pairwise = tv_summary.pairwise_distances;
            if size(current_pairwise, 2) ~= num_obs
                if size(current_pairwise, 1) == num_obs
                    current_pairwise = current_pairwise.';
                else
                    error('Unexpected pairwise_distances dimensions in %s.', results_file);
                end
            end
            current_vs_opt = tv_summary.vs_optimal_distances;
            if size(current_vs_opt, 2) ~= num_obs
                if size(current_vs_opt, 1) == num_obs
                    current_vs_opt = current_vs_opt.';
                else
                    error('Unexpected vs_optimal_distances dimensions in %s.', results_file);
                end
            end

            if isempty(tv_pair_runs)
                pairwise_labels = tv_summary.pairwise_labels(:);
                method_labels = tv_summary.method_labels(:).';
                optimal_label = tv_summary.optimal_label;
                num_methods = numel(method_labels);
                num_pairs = size(current_pairwise, 1);
                tv_pair_runs = zeros(num_runs, num_pairs, num_obs);
                tv_vs_opt_runs = zeros(num_runs, num_methods, num_obs);
            else
                if numel(tv_summary.method_labels) ~= num_methods
                    error('Method label count changed between runs.');
                end
                if size(current_pairwise, 1) ~= size(tv_pair_runs, 2)
                    error('Pairwise distance count changed between runs.');
                end
            end

            tv_pair_runs(run_idx, :, :) = current_pairwise;
            tv_vs_opt_runs(run_idx, :, :) = current_vs_opt;
        else
            required_tv_fields = {'inside_AB', 'inside_A_vs_optimal', 'inside_B_vs_optimal'};
            for k = 1:numel(required_tv_fields)
                if ~isfield(tv_summary, required_tv_fields{k})
                    error('tv_summary.%s missing from %s.', required_tv_fields{k}, results_file);
                end
            end

            if isempty(tv_AB_runs)
                tv_AB_runs = zeros(num_runs, num_obs);
                tv_A_opt_runs = zeros(num_runs, num_obs);
                tv_B_opt_runs = zeros(num_runs, num_obs);
            end

            tv_AB_runs(run_idx, :) = tv_summary.inside_AB(:).';
            tv_A_opt_runs(run_idx, :) = tv_summary.inside_A_vs_optimal(:).';
            tv_B_opt_runs(run_idx, :) = tv_summary.inside_B_vs_optimal(:).';
        end

        if using_new_mse
            current_vs_opt_mse = mse_summary.vs_optimal;
            if size(current_vs_opt_mse, 2) ~= num_obs
                if size(current_vs_opt_mse, 1) == num_obs
                    current_vs_opt_mse = current_vs_opt_mse.';
                else
                    error('Unexpected mse_summary.vs_optimal dimensions in %s.', results_file);
                end
            end
            current_signal_mse = mse_summary.signal;
            if size(current_signal_mse, 2) ~= num_obs
                if size(current_signal_mse, 1) == num_obs
                    current_signal_mse = current_signal_mse.';
                else
                    error('Unexpected mse_summary.signal dimensions in %s.', results_file);
                end
            end
            current_signal_opt = mse_summary.signal_optimal(:).';

            if isempty(mse_vs_opt_runs)
                mse_vs_opt_runs = zeros(num_runs, size(current_vs_opt_mse, 1), num_obs);
                mse_signal_runs = zeros(num_runs, size(current_signal_mse, 1), num_obs);
                mse_signal_opt_runs = zeros(num_runs, num_obs);
            else
                if size(current_vs_opt_mse, 1) ~= size(mse_vs_opt_runs, 2)
                    error('MSE method count changed between runs.');
                end
                if size(current_signal_mse, 1) ~= size(mse_signal_runs, 2)
                    error('Signal-referenced MSE method count changed between runs.');
                end
            end

            mse_vs_opt_runs(run_idx, :, :) = current_vs_opt_mse;
            mse_signal_runs(run_idx, :, :) = current_signal_mse;
            mse_signal_opt_runs(run_idx, :) = current_signal_opt;
        else
            required_mse_fields = {'methodA', 'methodB', 'MSE_signal_A', 'MSE_signal_B'};
            for k = 1:numel(required_mse_fields)
                if ~isfield(mse_summary, required_mse_fields{k})
                    error('mse_summary.%s missing from %s.', required_mse_fields{k}, results_file);
                end
            end

            if isempty(mse_A_runs)
                mse_A_runs = zeros(num_runs, num_obs);
                mse_B_runs = zeros(num_runs, num_obs);
                mse_signal_A_runs = zeros(num_runs, num_obs);
                mse_signal_B_runs = zeros(num_runs, num_obs);
            end

            mse_A_runs(run_idx, :) = mse_summary.methodA(:).';
            mse_B_runs(run_idx, :) = mse_summary.methodB(:).';
            mse_signal_A_runs(run_idx, :) = mse_summary.MSE_signal_A(:).';
            mse_signal_B_runs(run_idx, :) = mse_summary.MSE_signal_B(:).';
        end
    end

    if using_new_tv
        num_pairs = size(tv_pair_runs, 2);
        num_methods = size(tv_vs_opt_runs, 2);

        tv_AB_runs = reshape(tv_pair_runs(:, 1, :), num_runs, num_obs);
        tv_A_opt_runs = reshape(tv_vs_opt_runs(:, 1, :), num_runs, num_obs);
        tv_B_opt_runs = reshape(tv_vs_opt_runs(:, 2, :), num_runs, num_obs);

        mean_tv_pairwise = squeeze(mean(tv_pair_runs, 1));
        mean_tv_vs_opt = squeeze(mean(tv_vs_opt_runs, 1));
        mean_tv_AB = mean(tv_AB_runs, 1);
        mean_tv_A_opt = mean(tv_A_opt_runs, 1);
        mean_tv_B_opt = mean(tv_B_opt_runs, 1);

        mean_mse_vs_opt = squeeze(mean(mse_vs_opt_runs, 1));
        mean_mse_signal_methods = squeeze(mean(mse_signal_runs, 1));
        mean_mse_signal_opt = mean(mse_signal_opt_runs, 1);

        mse_A_runs = reshape(mse_vs_opt_runs(:, 1, :), num_runs, num_obs);
        mse_B_runs = reshape(mse_vs_opt_runs(:, 2, :), num_runs, num_obs);
        mse_signal_A_runs = reshape(mse_signal_runs(:, 1, :), num_runs, num_obs);
        mse_signal_B_runs = reshape(mse_signal_runs(:, 2, :), num_runs, num_obs);
        mean_mse_A = mean(mse_A_runs, 1);
        mean_mse_B = mean(mse_B_runs, 1);
        mean_mse_signal_A = mean(mse_signal_A_runs, 1);
        mean_mse_signal_B = mean(mse_signal_B_runs, 1);

        colors_runs = lines(num_runs);
        method_colors = lines(num_methods);
        pair_colors = lines(max(num_pairs, 1));
        line_styles = {'-', '--', ':', '-.'};
        if num_methods > numel(line_styles)
            line_styles = repmat(line_styles, 1, ceil(num_methods / numel(line_styles)));
        end

        fig_runs_ab = figure('Name', 'Posterior TV distance trajectories (Method A vs B)');
        hold on;
        for run_idx = 1:num_runs
            plot(obs_indices, tv_AB_runs(run_idx, :), 'Color', colors_runs(run_idx, :), ...
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
        legend_added = false(1, num_methods);
        for method_idx = 1:num_methods
            for run_idx = 1:num_runs
                line_handle = plot(obs_indices, squeeze(tv_vs_opt_runs(run_idx, method_idx, :)).', ...
                    'Color', method_colors(method_idx, :), 'LineStyle', line_styles{method_idx}, ...
                    'LineWidth', 1.0);
                if ~legend_added(method_idx)
                    set(line_handle, 'DisplayName', sprintf('%s vs %s', method_labels{method_idx}, optimal_label));
                    legend_added(method_idx) = true;
                else
                    set(line_handle, 'HandleVisibility', 'off');
                end
            end
        end
        hold off;
        xlabel('Observation index');
        ylabel('TV distance vs optimal');
        title('TV distances between SIR optimal filter and barrier methods');
        legend('Location', 'bestoutside');
        grid on;

        fig_mean_tv = figure('Name', 'Average posterior TV distances (pairwise)');
        hold on;
        for pair_idx = 1:num_pairs
            plot(obs_indices, mean_tv_pairwise(pair_idx, :), 'LineWidth', 1.5, ...
                'Color', pair_colors(pair_idx, :), 'DisplayName', pairwise_labels{pair_idx});
        end
        hold off;
        xlabel('Observation index');
        ylabel('TV distance');
        title('Average posterior TV distances across simulations');
        legend('Location', 'best');
        grid on;

        fig_mean_vs_opt = figure('Name', 'Average posterior TV distances vs optimal');
        hold on;
        for method_idx = 1:num_methods
            plot(obs_indices, mean_tv_vs_opt(method_idx, :), 'LineWidth', 1.5, ...
                'Color', method_colors(method_idx, :), 'LineStyle', line_styles{method_idx}, ...
                'DisplayName', sprintf('%s vs %s', method_labels{method_idx}, optimal_label));
        end
        hold off;
        xlabel('Observation index');
        ylabel('TV distance');
        title('Average TV distances between barrier methods and optimal filter');
        legend('Location', 'best');
        grid on;

        fig_mean_mse = figure('Name', 'Average normalized filtered MSE');
        hold on;
        for method_idx = 1:num_methods
            plot(obs_indices, mean_mse_vs_opt(method_idx, :), 'LineWidth', 1.5, ...
                'Color', method_colors(method_idx, :), 'LineStyle', line_styles{method_idx}, ...
                'DisplayName', method_labels{method_idx});
        end
        hold off;
        xlabel('Observation index');
        ylabel('Normalized MSE');
        title('Average normalized filtered MSE vs optimal filter');
        legend('Location', 'best');
        grid on;

        fig_mean_signal_mse = figure('Name', 'Average signal-referenced MSE');
        hold on;
        for method_idx = 1:num_methods
            plot(obs_indices, mean_mse_signal_methods(method_idx, :), 'LineWidth', 1.5, ...
                'Color', method_colors(method_idx, :), 'LineStyle', line_styles{method_idx}, ...
                'DisplayName', sprintf('%s vs signal', method_labels{method_idx}));
        end
        plot(obs_indices, mean_mse_signal_opt, 'k--', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%s vs signal', optimal_label));
        hold off;
        xlabel('Observation index');
        ylabel('Normalized MSE');
        title('Average signal-referenced MSE across simulations');
        legend('Location', 'best');
        grid on;

        stats_file = fullfile(base_output_dir, 'tv_distance_statistics.mat');
        save(stats_file, 'obs_indices', 'pairwise_labels', 'method_labels', 'optimal_label', ...
            'tv_pair_runs', 'tv_vs_opt_runs', 'tv_AB_runs', 'tv_A_opt_runs', 'tv_B_opt_runs', ...
            'mean_tv_pairwise', 'mean_tv_vs_opt', 'mean_tv_AB', 'mean_tv_A_opt', 'mean_tv_B_opt', ...
            'mse_vs_opt_runs', 'mse_signal_runs', 'mse_signal_opt_runs', ...
            'mse_A_runs', 'mse_B_runs', 'mse_signal_A_runs', 'mse_signal_B_runs', ...
            'mean_mse_vs_opt', 'mean_mse_signal_methods', 'mean_mse_signal_opt', ...
            'mean_mse_A', 'mean_mse_B', 'mean_mse_signal_A', 'mean_mse_signal_B');
        fprintf('Saved aggregated statistics to %s\n', stats_file);

        saveas(fig_runs_ab, fullfile(base_output_dir, 'tv_distance_runs_AB.fig'));
        saveas(fig_runs_opt, fullfile(base_output_dir, 'tv_distance_runs_vs_optimal.fig'));
        saveas(fig_mean_tv, fullfile(base_output_dir, 'tv_distance_average.fig'));
        saveas(fig_mean_vs_opt, fullfile(base_output_dir, 'tv_distance_average_vs_optimal.fig'));
        saveas(fig_mean_mse, fullfile(base_output_dir, 'filtered_mse_average.fig'));
        saveas(fig_mean_signal_mse, fullfile(base_output_dir, 'signal_mse_average.fig'));

        if nargout > 0
            stats = struct(...
                'obs_indices', obs_indices, ...
                'pairwise_labels', {pairwise_labels}, ...
                'method_labels', {method_labels}, ...
                'optimal_label', optimal_label, ...
                'tv_pair_runs', tv_pair_runs, ...
                'tv_vs_opt_runs', tv_vs_opt_runs, ...
                'tv_AB_runs', tv_AB_runs, ...
                'tv_A_opt_runs', tv_A_opt_runs, ...
                'tv_B_opt_runs', tv_B_opt_runs, ...
                'mean_tv_pairwise', mean_tv_pairwise, ...
                'mean_tv_vs_opt', mean_tv_vs_opt, ...
                'mean_tv_AB', mean_tv_AB, ...
                'mean_tv_A_opt', mean_tv_A_opt, ...
                'mean_tv_B_opt', mean_tv_B_opt, ...
                'mse_vs_opt_runs', mse_vs_opt_runs, ...
                'mse_signal_runs', mse_signal_runs, ...
                'mse_signal_opt_runs', mse_signal_opt_runs, ...
                'mse_A_runs', mse_A_runs, ...
                'mse_B_runs', mse_B_runs, ...
                'mse_signal_A_runs', mse_signal_A_runs, ...
                'mse_signal_B_runs', mse_signal_B_runs, ...
                'mean_mse_vs_opt', mean_mse_vs_opt, ...
                'mean_mse_signal_methods', mean_mse_signal_methods, ...
                'mean_mse_signal_opt', mean_mse_signal_opt, ...
                'mean_mse_A', mean_mse_A, ...
                'mean_mse_B', mean_mse_B, ...
                'mean_mse_signal_A', mean_mse_signal_A, ...
                'mean_mse_signal_B', mean_mse_signal_B, ...
                'base_output_dir', base_output_dir, ...
                'run_dirs', {fullfile(base_output_dir, {run_listing.name})});
        end
    else
        mean_tv_AB = mean(tv_AB_runs, 1);
        mean_tv_A_opt = mean(tv_A_opt_runs, 1);
        mean_tv_B_opt = mean(tv_B_opt_runs, 1);
        mean_mse_A = mean(mse_A_runs, 1);
        mean_mse_B = mean(mse_B_runs, 1);
        mean_mse_signal_A = mean(mse_signal_A_runs, 1);
        mean_mse_signal_B = mean(mse_signal_B_runs, 1);

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

        fig_mean_signal_mse = figure('Name', 'Average signal-referenced MSE');
        hold on;
        plot(obs_indices, mean_mse_signal_A, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, ...
            'DisplayName', 'Method A vs signal');
        plot(obs_indices, mean_mse_signal_B, '-s', 'LineWidth', 1.5, 'MarkerSize', 6, ...
            'DisplayName', 'Method B vs signal');
        hold off;
        xlabel('Observation index');
        ylabel('Normalized MSE');
        title('Average signal-referenced MSE across simulations');
        legend('Location', 'best');
        grid on;

        stats_file = fullfile(base_output_dir, 'tv_distance_statistics.mat');
        save(stats_file, 'obs_indices', 'tv_AB_runs', 'tv_A_opt_runs', 'tv_B_opt_runs', ...
            'mean_tv_AB', 'mean_tv_A_opt', 'mean_tv_B_opt', 'mse_A_runs', 'mse_B_runs', ...
            'mse_signal_A_runs', 'mse_signal_B_runs', 'mean_mse_A', 'mean_mse_B', ...
            'mean_mse_signal_A', 'mean_mse_signal_B');
        fprintf('Saved aggregated statistics to %s\n', stats_file);

        saveas(fig_runs_ab, fullfile(base_output_dir, 'tv_distance_runs_AB.fig'));
        saveas(fig_runs_opt, fullfile(base_output_dir, 'tv_distance_runs_vs_optimal.fig'));
        saveas(fig_mean_tv, fullfile(base_output_dir, 'tv_distance_average.fig'));
        saveas(fig_mean_mse, fullfile(base_output_dir, 'filtered_mse_average.fig'));
        saveas(fig_mean_signal_mse, fullfile(base_output_dir, 'signal_mse_average.fig'));

        if nargout > 0
            stats = struct(...
                'obs_indices', obs_indices,...
                'tv_AB_runs', tv_AB_runs,...
                'tv_A_opt_runs', tv_A_opt_runs,...
                'tv_B_opt_runs', tv_B_opt_runs,...
                'mse_A_runs', mse_A_runs,...
                'mse_B_runs', mse_B_runs,...
                'mse_signal_A_runs', mse_signal_A_runs,...
                'mse_signal_B_runs', mse_signal_B_runs,...
                'mean_tv_AB', mean_tv_AB,...
                'mean_tv_A_opt', mean_tv_A_opt,...
                'mean_tv_B_opt', mean_tv_B_opt,...
                'mean_mse_A', mean_mse_A,...
                'mean_mse_B', mean_mse_B,...
                'mean_mse_signal_A', mean_mse_signal_A,...
                'mean_mse_signal_B', mean_mse_signal_B,...
                'base_output_dir', base_output_dir,...
                'run_dirs', {fullfile(base_output_dir, {run_listing.name})});
        end
    end
end
