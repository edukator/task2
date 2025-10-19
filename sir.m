function varargout = sir(F, sx, sz, h, NT, n_obs, z, H, X0, ness_thr, storage_opts)
%SIR Sequential-importance resampling particle filter.
%   [Xf, Xp, RESAMPLING_COUNTER] = SIR(F, SX, SZ, H, NT, N_OBS, Z, H, X0,
%   NESS_THR, STORAGE_OPTS) advances the particle ensemble by sequential
%   importance resampling. Optional STORAGE_OPTS.measurement_handler can be
%   supplied to stream particle summaries without persisting ensembles.
%
%   The function mirrors ``sir_barrier`` so that downstream analysis can
%   treat both filters uniformly. When called without output arguments the
%   filtered means are still computed internally, but the caller discards
%   them.

    if nargin < 11 || isempty(storage_opts)
        storage_opts = struct();
    end

    % Optional measurement handler invoked after each assimilation step.
    if isfield(storage_opts, 'measurement_handler') && ~isempty(storage_opts.measurement_handler)
        if ~isa(storage_opts.measurement_handler, 'function_handle')
            error('storage_opts.measurement_handler must be a function handle.');
        end
        measurement_handler = storage_opts.measurement_handler;
    else
        measurement_handler = [];
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

    W_history = [];
    Particle_history = [];

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
