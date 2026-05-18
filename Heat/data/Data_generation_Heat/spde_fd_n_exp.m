function [t, ut] = spde_fd_n_exp(u0, T, a, N, J, epsilon, sigma, ell, fhandle)
% Solves the stochastic heat equation using implicit Euler in time and
% finite differences in space:
%   du = epsilon * u_xx * dt + sigma * dW(t,x)
% Essentially a data generator.
% Inputs:
%   u0      - initial condition vector (J+1 x 1)
%   T       - end time
%   a       - spatial domain length (x goes from 0 to a)
%   N       - number of time steps
%   J       - number of spatial intervals (so J+1 grid points)
%   epsilon - diffusion coefficient [not for the actual diffusion term]
%   sigma   - noise amplitude
%   ell     - spatial correlation length of the noise
%   fhandle - function handle for nonlinear drift term f(u), e.g. @(u) 0
%
% Outputs:
%   t  - time vector (N+1 x 1)
%   ut - solution array (J+1 x N+1), each column is u(x) at one time step

% --- Time and space step sizes ---
Dt = T/N;           % time step size, e.g. 1/500 = 0.002
t  = [0:Dt:T]';     % time vector from 0 to T, length N+1
h  = a/J;           % spatial step size, e.g. 20/64 = 0.3125

% --- Build the tridiagonal finite difference Laplacian matrix A ---
% A approximates the second spatial derivative: A/h^2 ≈ d²/dx²
% Each interior row looks like: [1, -2, 1] / h^2
e = ones(J+1, 1);
A = spdiags([e -2*e e], -1:1, J+1, J+1); %specifically for heat systems. This essentially forces the drift to contain a diffusion term.
% A is (J+1)x(J+1) sparse tridiagonal:
%   main diagonal:  -2
%   off-diagonals:   1

% --- Apply Neumann (zero-flux) boundary conditions --- this ensures that there's nothing changing at the boundaries.
% Modifying corners using the ghost-point method so that du/dx = 0 at
% both x=0 and x=a. The factor of 2 replaces the off-diagonal 1
% to account for the reflected ghost point.
ind = 1:J+1;        % index of all spatial grid points
A(1,2)       = 2;   % left boundary: x = 0
A(end,end-1) = 2;   % right boundary: x = a.

% --- Build the implicit Euler system matrix ---
% Rearranging the implicit Euler step:
%   (I - Dt*epsilon*A/h^2) * u_{n+1} = u_n + Dt*f(u_n) + sigma*sqrt(Dt)*dW
% We precompute the left-hand side matrix once since it never changes.
EE = speye(length(ind)) - Dt*epsilon*A/h/h;

% --- Initialise the solution array ---
ut      = zeros(J+1, length(t));   % preallocate: rows=space, cols=time
ut(:,1) = u0;                      % set first column to initial condition
u_n     = u0(ind);                 % current state vector (will be updated)

% --- Noise generation flag ---
% circulant_exp generates TWO noise samples per call (dW and dW2).
% We alternate between them to avoid calling the generator every step.
% flag=false means we need to generate a fresh pair.
flag = false;

% --- Time loop ---
for k = 1:N

    % Evaluate the nonlinear drift term at current state
    % (always zero here since fhandle = @(u) 0, but kept for generality)
    fu = fhandle(u_n);

    % --- Generate spatially correlated noise ---
    if flag == false
        % Generate two independent noise samples dW and dW2
        [x, dW, dW2] = circulant_exp(length(ind), h, ell);
        flag = true;    % use dW2 on the next step
    else
        % Reuse the second sample from the previous call
        dW   = dW2;
        flag = false;   % generate a fresh pair on the next step
    end

    % --- Implicit Euler update ---
    % Solve: EE * u_{n+1} = u_n + Dt*f(u_n) + sigma*sqrt(Dt)*dW
    % The \ operator solves the linear system (inverts EE efficiently
    % as it is a sparse tridiagonal matrix).
    % sigma*sqrt(Dt)*dW is the discretised Ito noise increment.
    u_new = EE \ (u_n + Dt*fu + sigma*sqrt(Dt)*dW);

    % --- Store result and advance ---
    ut(ind, k+1) = u_new;   % store new solution in next column of ut
    u_n          = u_new;   % update current state for next iteration

end