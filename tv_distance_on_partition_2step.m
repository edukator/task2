function [Dhat, details] = tv_distance_on_partition_2step(Xa, wa, Xb, wb, S)
% [Dhat, details] = tv_distance_on_partition_2step(Xa, wa, Xb, wb, S)
% Xa, Xb : d×N_a / d×N_b (columns = particles)
% wa, wb : N_a×1 / N_b×1 weights (need not be normalized)
% S      : struct from hypercube_partition_full (uses parent_bounds, sub_centers, r_sub)
%
% Output:
%   Dhat    : scalar TV distance for the inside mass only
%   details : struct with fields:
%       .A_main, .B_main        : outputs of maincube_mass for A and B
%       .masses_A, .masses_B    : M×1 conditional masses on subcubes
%       .inside_masses_A, .inside_masses_B : actual probability masses inside
%       .D_inside               : 0.5 * sum |inside_masses_A - inside_masses_B|
%       .D_conditional          : 0.5 * sum |masses_A - masses_B|

    % Step 1: main-cube masses and inside subsets
    A = maincube_mass(Xa, wa, S);   % gives X_in, w_in, inside/outside mass&counts
    B = maincube_mass(Xb, wb, S);

    % Step 2: subcube masses using only inside particles (each renormalized to 1)
    mA = subcube_masses_loop(A.X_in, A.w_in, S);   % M×1
    mB = subcube_masses_loop(B.X_in, B.w_in, S);   % M×1

    % TV on cells (conditional inside distributions)
    D_conditional = 0.5 * sum(abs(mA - mB));

    % Re-scale by the probability mass that actually falls inside the cube
    inside_masses_A = A.inside_mass * mA;
    inside_masses_B = B.inside_mass * mB;
    D_inside = 0.5 * sum(abs(inside_masses_A - inside_masses_B));

    Dhat = D_inside;

    details = struct( ...
        'A_main', A, ...
        'B_main', B, ...
        'masses_A', mA, ...
        'masses_B', mB, ...
        'inside_masses_A', inside_masses_A, ...
        'inside_masses_B', inside_masses_B, ...
        'D_inside', D_inside, ...
        'D_conditional', D_conditional);
end
