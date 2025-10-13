function S = hypercube_partition_full(cent, r, r_sub)
% S = hypercube_partition_full(cent, r, r_sub)
% cent : d×1 center vector in R^d
% r    : scalar, semi-length of the parent d-D hypercube
% r_sub: scalar, semi-length of the sub-hypercubes
%
% Partitions the full d-D hypercube [cent - r, cent + r] into identical
% sub-hypercubes of semi-length r_sub (edge length 2*r_sub) along ALL axes.
%
% Returns struct S with:
%   .d, .r, .r_sub, .K
%   .parent_bounds : 2 x d (low; high)
%   .grid_axes     : 1 x d cell, each with K centers
%   .lin_idx_shape : 1 x d, all = K
%   .sub_centers   : d x (K^d) matrix, each column is a sub-center
%   .sub_index_nd  : (K^d) x d integer grid indices (1..K per axis)
%   .sub_bounds    : (K^d) x 2 x d (low/high per axis)
%   .vertices_fn   : handle @(i) -> d x (2^d) matrix of vertices of i-th subcube

    assert(iscolumn(cent), 'cent must be a column vector (d×1).');
    d = numel(cent);

    Kf = r / r_sub;
    K  = round(Kf);
    tol = 1e-12 * max(1, abs(r));
    assert(abs(Kf - K) < tol, 'r must be an integer multiple of r_sub.');

    offsets_1d = (-K+1 : 2 : K-1) * r_sub;
    grid_axes = cell(1, d);
    for j = 1:d
        grid_axes{j} = cent(j) + offsets_1d;
    end

    [G{1:d}] = ndgrid(grid_axes{:});
    total = K^d;

    sub_centers_mat = zeros(d, total);
    sub_index_nd = zeros(total, d);
    for j = 1:d
        Gj = G{j};
        vals = Gj(:);
        sub_centers_mat(j, :) = vals.';
        [~, idxj] = ismember(vals, grid_axes{j});
        sub_index_nd(:, j) = idxj;
    end

    low  = sub_centers_mat.' - r_sub;
    high = sub_centers_mat.' + r_sub;
    sub_bounds = reshape(cat(2, low, high), total, 2, d);

    parent_bounds = [cent.' - r; cent.' + r];

    vertices_fn = @(i) local_vertices_full(sub_centers_mat(:, i), r_sub);

    S.d              = d;
    S.r              = r;
    S.r_sub          = r_sub;
    S.K              = K;
    S.parent_bounds  = parent_bounds;
    S.grid_axes      = grid_axes;
    S.lin_idx_shape  = K * ones(1, d);
    S.sub_centers    = sub_centers_mat;
    S.sub_index_nd   = sub_index_nd;
    S.sub_bounds     = sub_bounds;
    S.vertices_fn    = vertices_fn;
end

function V = local_vertices_full(c_sub, r_sub)
% Vertices as d x (2^d), each column is a vertex
    d = numel(c_sub);
    B = dec2bin(0:(2^d - 1)) - '0';
    signs = 2*B - 1;
    V = c_sub + r_sub * signs.';
end

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

    w = w(:);
    if any(w < 0) || ~isfinite(sum(w)) || sum(w) <= 0
        error('Weights must be nonnegative with positive finite sum.');
    end
    w_pdf = w / sum(w);

    low  = S.parent_bounds(1,:).';
    high = S.parent_bounds(2,:).';

    idx_inside  = all((X >= low) & (X <= high), 1);
    idx_outside = find(~idx_inside);

    inside_count  = sum(idx_inside);
    outside_count = numel(idx_outside);

    inside_mass  = sum(w_pdf(idx_inside));
    outside_mass = 1 - inside_mass;

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