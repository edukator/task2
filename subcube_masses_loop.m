function masses = subcube_masses_loop(X_in, w_in, S)
% masses = subcube_masses_loop(X_in, w_in, S)
% X_in : d×N_in  (assumed INSIDE the main cube)
% w_in : N_in×1  (weights, will be normalized here for safety)
% S    : struct from hypercube_partition_full
%
% Output:
%   masses : M×1 vector of probabilities on each sub-hypercube, M = K^d
%            (sums to 1 up to numerical roundoff)

    w_in = w_in(:);
    if isempty(w_in)
        masses = zeros(size(S.sub_centers,2),1);
        return;
    end
    w_in = w_in / sum(w_in);          % normalize inside weights

    r_sub = S.r_sub;
    C     = S.sub_centers;            % d×M (columns are sub-centers)
    M     = size(C,2);

    masses = zeros(M,1);
    for j = 1:M
        c  = C(:,j);
        in_cell = all(abs(X_in - c) <= r_sub, 1);   % particle in j-th subcube
        masses(j) = w_in.' * in_cell(:);
    end
end
