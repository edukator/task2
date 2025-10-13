

function [Xf, Xp,resampling_counter] = sir(F,sx,sz,h,NT,n_obs,z,H,X0,ness_thr)
% function [Xf, Xp] = sir(F,sx,sz,h,NT,n_obs,z,X0,ness_thr)
%
% F : Lorenz 96 forcing parameter
% sx : process noise std
% sz : observation noise std
% h : Euler integration step
% NT : no. of discrete time steps
% n_obs : observations are collected every n_obs discrete time units
% z : observations
% H : observation matrix
% X0 : initial particles 
% ness_thr : resampling threshold
% 
% Xf : filtered states
% Xp : predicted states
%
% Standard particle filter, resampling conditional on ESS
%

% recovers the no. of particles (N), the no. of slow oscillators (nosc)
[Dx, N] = size(X0);
nt=NT/n_obs; % choose NT and obs such that obs| NT
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
sxsqrth=sx*sqrt(h);
 
%
% --Time loop
%
resampling_counter=0;
%finer_idx_counter=1;
for obs_idx=1:nt

   obs_z=z(:,obs_idx); %  i eliminated zeros, z(1) gives observed value at t=h 
  

   %fprintf("observation index  of   %d  at time  , \n ",obs_idx);
   for inner_idx=1:n_obs
        
        Xdrift = l96dxdt(Xold,F,Dx);
        dWx=sxsqrth*randn(Dx,N);
       % Xdrift = Xdrift./(1+h*vecnorm(Xdrift)); % example taming function
        Xnew=  Xold + h*Xdrift+dWx;
        %finer_idx_counter=finer_idx_counter+1;

        % prediction
        Xold=Xnew;
        Xp(:,(obs_idx-1)*n_obs+inner_idx+1)=Xnew*w';
       % fprintf("finer_idx  % d, Xp stored at   %d \n", finer_idx_counter,(obs_idx-1)*n_obs+inner_idx+1);
   end 
   
   

   % prediction
     
       % Available observations

       % weights
       llk = -(1/(2*s2z)) .* sum(( obs_z - H*Xnew ).^2);  % log-likelihood
       lw = lw + llk - max(lw+llk);
       wu = exp(lw);
       w = wu ./ sum(wu);
       
       % estimation
       Xf(:,obs_idx+1) = Xnew*w';  %%% is it the correct place ?  (obs_idx+1)
       
       % fprintf("finer_idx  % d, Xp stored at   %d \n", finer_idx_counter,(obs_idx-1)*n_obs+inner_idx+1);     
       % Resampling
       NESS = (1/sum(w.^2))/N; 
       if NESS<ness_thr
            idx = randsample(1:N, N, true, w);
            Xnew(:,1:N) = Xnew(:,idx);
            Xold = Xnew;
            w = ones([1 N])/N;
            lw = zeros([1 N]);
            resampling_counter=resampling_counter+1;
       end %if
       
   %fprintf("--------------\n");
    
   % ...for the next time step
  % Xold = Xnew;
    
end % time (n)
