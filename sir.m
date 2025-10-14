function varargout = sir(F, sx, sz, h, NT, n_obs, z, H, X0, ness_thr, storage_opts)
%SIR Sequential-importance resampling particle filter.
%   [Xf, Xp, RESAMPLING_COUNTER] = SIR(F, SX, SZ, H, NT, N_OBS, Z, H, X0,
%   NESS_THR, STORAGE_OPTS) advances the particle ensemble by sequential
%   importance resampling. Optional STORAGE_OPTS control whether posterior
%   ensembles are saved to disk or streamed via callbacks so that particles
%   and weights need not be stored persistently.
%
%   STORAGE_OPTS fields (all optional):
%       save_to_disk         - Logical flag. When true, particles are stored
%                              under STORAGE_OPTS.output_dir/METHOD_LABEL.
%       method_label         - Identifier used for disk storage (required
%                              when save_to_disk is true).
%       output_dir           - Directory for posterior data (defaults to a
%                              ``posterior_data`` folder in the current
%                              directory).
%       measurement_handler  - Function handle called as
%                              handler(obs_idx, particles, weights) after
%                              every assimilation step.
%       store_histories      - When true and save_to_disk is false, particle
%                              ensembles and weights are retained in memory.
%
%   The function mirrors ``sir_barrier`` so that downstream analysis can
%   treat both filters uniformly. When called without output arguments the
%   filtered means are still computed internally, but the caller discards
%   them.

    if nargin < 11 || isempty(storage_opts)
        storage_opts = struct();
    end

    save_to_disk = isfield(storage_opts, 'save_to_disk') && storage_opts.save_to_disk;

    if isfield(storage_opts, 'measurement_handler') && ~isempty(storage_opts.measurement_handler)
        if ~isa(storage_opts.measurement_handler, 'function_handle')
            error('storage_opts.measurement_handler must be a function handle.');
        end
        measurement_handler = storage_opts.measurement_handler;
    else
        measurement_handler = [];
    end

    if isfield(storage_opts, 'store_histories') && ~save_to_disk
        store_histories = logical(storage_opts.store_histories);
    else
        store_histories = ~save_to_disk;  % mimic legacy behaviour
    end

    if save_to_disk
        if ~isfield(storage_opts, 'method_label') || isempty(storage_opts.method_label)
            error('When save_to_disk=true, storage_opts.method_label must be provided.');
        end
        method_label = storage_opts.method_label;

        if isfield(storage_opts, 'output_dir') && ~isempty(storage_opts.output_dir)
            output_dir = storage_opts.output_dir;
        else
            output_dir = fullfile(pwd, 'posterior_data');
        end

        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end

        method_dir = fullfile(output_dir, method_label);
        if exist(method_dir, 'dir')
            existing_files = dir(fullfile(method_dir, '*.mat'));
            if ~isempty(existing_files)
                delete(fullfile(method_dir, '*.mat'));
            end
        else
            mkdir(method_dir);
        end

        summary_file = fullfile(method_dir, 'summary.mat');
    else
        method_label = '';
        method_dir = '';
        summary_file = '';
    end

    [Dx, N] = size(X0);
    nt = NT/n_obs; % choose NT and obs such that obs|NT

    % initialisation
    Xf = zeros([Dx nt+1]);      % filtered estimates, at obs
    Xp = zeros([Dx NT+1]);      % predicted estimates, at all times
    Xf(:,1) = mean(X0,2);
    Xp(:,1) = mean(X0,2);
    Xold = X0;                % auxiliary
    lw = zeros([1 N]);        % log-weights
    w = ones([1 N])/N;        % weights

    % noise variance
    s2z = sz^2;
    % time steps
    sxsqrth = sx*sqrt(h);

    if save_to_disk || ~store_histories
        W_history = [];
        Particle_history = [];
    else
        W_history   = cell(nt, 1);
        Particle_history = zeros([Dx, N, nt]);
    end

    resampling_counter = 0;

    for obs_idx = 1:nt
        obs_z = z(:,obs_idx); % observation at current time

        for inner_idx = 1:n_obs
            Xdrift = l96dxdt(Xold,F,Dx);
            dWx = sxsqrth*randn(Dx,N);
            Xnew =  Xold + h*Xdrift+dWx;

            % prediction
            Xold = Xnew;
            Xp(:,(obs_idx-1)*n_obs+inner_idx+1) = Xnew*w';
        end

        % weights update
        llk = -(1/(2*s2z)) .* sum(( obs_z - H*Xnew ).^2);  % log-likelihood
        lw = lw + llk - max(lw+llk);
        wu = exp(lw);
        w = wu ./ sum(wu);

        if ~isempty(measurement_handler)
            measurement_handler(obs_idx, Xnew, w);
        end

        if save_to_disk
            obs_data = struct();
            obs_data.method_label = method_label;
            obs_data.observation_index = obs_idx;
            obs_data.particles = Xnew;
            obs_data.weights = w;
            obs_file = fullfile(method_dir, sprintf('obs_%04d.mat', obs_idx));
            save(obs_file, '-struct', 'obs_data', '-v7.3');
        elseif store_histories
            Particle_history(:,:,obs_idx) = Xnew;
            W_history{obs_idx} = w;
        end

        % estimation
        Xf(:,obs_idx+1) = Xnew*w';

        % Resampling
        NESS = (1/sum(w.^2))/N;
        if NESS<ness_thr
            idx = randsample(1:N, N, true, w);
            Xnew(:,1:N) = Xnew(:,idx);
            Xold = Xnew;
            w = ones([1 N])/N;
            lw = zeros([1 N]);
            resampling_counter = resampling_counter+1;
        end
    end % time (n)

    if save_to_disk
        summary = struct();
        summary.method_label = method_label;
        summary.output_dir = method_dir;
        summary.resampling_counter = resampling_counter;
        summary.n_obs = n_obs;
        summary.NT = NT;
        summary.Dx = Dx;
        summary.N = N;
        save(summary_file, '-struct', 'summary');
    end

    if nargout > 0
        base_outputs = {Xf, Xp, resampling_counter, W_history, Particle_history};
        nout = min(nargout, numel(base_outputs));
        varargout = cell(1, nargout);
        for k = 1:nout
            varargout{k} = base_outputs{k};
        end
        for k = (nout+1):nargout
            varargout{k} = [];
        end
    end
end
