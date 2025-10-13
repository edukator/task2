function R = maincube_mass(X, w, S)
% R = maincube_mass(X, w, S)
% X : d×N  (columns = particles)
% w : N×1  (nonnegative weights; need not be normalized)
% S : struct from hypercube_partition_full (uses S.parent_bounds)
%
% Returns R with:
%   .inside_mass     : sum of weights INSIDE the main cube (probability)
%   .outside_mass    : 1 - inside_mass
%   .inside_count    : number of particles inside
%   .outside_count   : number of particles outside
%   .inside_fraction : inside_count / N
%   .idx_inside      : 1×N logical
%   .idx_outside     : indices of outside particles
%   .X_in            : d×N_in   inside particles
%   .w_in            : N_in×1   inside weights, normalized to sum 1
%
% Notes:
% - If no particles are inside, X_in is d×0 and w_in is 0×1.

    % --- validate & normalize original weights to a pdf
    w = w(:);
    if any(w < 0) || ~isfinite(sum(w)) || sum(w) <= 0
        error('Weights must be nonnegative with positive finite sum.');
    end
    w_pdf = w / sum(w);

    % --- main cube bounds
    low  = S.parent_bounds(1,:).';   % d×1
    high = S.parent_bounds(2,:).';   % d×1

    % --- who is inside the main cube?
    idx_inside  = all((X >= low) & (X <= high), 1);
    idx_outside = find(~idx_inside);

    inside_count  = sum(idx_inside);
    outside_count = numel(idx_outside);

    inside_mass  = sum(w_pdf(idx_inside));
    outside_mass = 1 - inside_mass;

    % --- extract inside subset and renormalize its weights to 1
    X_in = X(:, idx_inside);
    if inside_mass > 0
        w_in = w_pdf(idx_inside) / inside_mass;
    else
        w_in = zeros(0,1);
    end

    R = struct( ...
        'inside_mass',  inside_mass, ...
        'outside_mass', outside_mass, ...
        'inside_count', inside_count, ...
        'outside_count', outside_count, ...
        'inside_fraction', inside_count / size(X,2), ...
        'idx_inside',  idx_inside, ...
        'idx_outside', idx_outside, ...
        'X_in', X_in, ...
        'w_in', w_in);
end
