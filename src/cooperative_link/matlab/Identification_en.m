%% Trajectory Matching and Teammate Identification (Vector-Direction Constrained)
% -------------------------------------------------------------------------
% Description: 
%   This script implements an iterative trajectory alignment algorithm 
%   incorporating Geometric, Temporal, and Vector-Directional constraints.
%   It uses Dynamic Time Warping (DTW) for correspondence and Weighted SVD 
%   for rigid transformation estimation.
% -------------------------------------------------------------------------

clear; clc; close all;
rng(1); % For reproducibility

% 仅保留学术字体
set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultTextFontName','Times New Roman');

%% 1. Scenario Simulation
N = 50; M = 50; dt = 0.2; noise_lvl = 0.05;

t_range = linspace(0, pi, N);
TS = [cumsum(cos(t_range)*0.5); cumsum(sin(t_range)*0.5)] * dt;
TS = TS + noise_lvl * randn(2,N);

theta_true = deg2rad(30); 
R_true = [cos(theta_true) -sin(theta_true); sin(theta_true) cos(theta_true)];
t_true = [5; 2];
TD1 = R_true * (TS - t_true) + noise_lvl * randn(2,N);

TD2 = [linspace(-2, -4, M); linspace(-1.5, 3.5, M)] + cos(t_range)*2 + noise_lvl*randn(2,M);

candidates = {TD1, TD2};
tau_s = linspace(0, 1, N);
tau_d = linspace(0, 1, M);

%% 2. Algorithm Parameters
params.lambda_t = 2.0;      
params.lambda_v = 3.5;      
params.dtw_window = 0.3;    
params.skip_penalty = 1.2;  
params.max_iter = 50;
params.tol = 1e-6;

%% 3. Core Identification Logic
results = struct();
for m = 1:length(candidates)
    TD = candidates{m};
    t = mean(TS, 2) - mean(TD, 2); 
    R = eye(2); 
    
    Vs = diff(TS, 1, 2); Vs = [Vs, Vs(:,end)]; 
    Vd = diff(TD, 1, 2); Vd = [Vd, Vd(:,end)];
    
    last_cost = inf; norm_geom = 1.0; 
    fprintf('Processing Candidate %d...\n', m);
    
    for iter = 1:params.max_iter
        TD_trans = R * TD + t;
        C_geom = pdist2(TS', TD_trans').^2;
        C_time = abs(repmat(tau_s', 1, M) - repmat(tau_d, N, 1));
        
        Vd_rot = R * Vd;
        C_vec_unit = 1 - sum(Vs .* Vd_rot) ./ (sqrt(sum(Vs.^2)).*sqrt(sum(Vd_rot.^2)) + 1e-6);
        C_vector = repmat(C_vec_unit, N, 1); 

        if iter == 1
            norm_geom = max(mean(C_geom(:)), 0.1);
        end
        
        C = (C_geom/norm_geom) + params.lambda_t * C_time + params.lambda_v * C_vector;
        mask = (C_time <= params.dtw_window);
        C_dtw = C; C_dtw(~mask) = max(C(:)) + 10;
        
        D = inf(N, M); Prev = zeros(N, M); D(1,1) = C_dtw(1,1);
        for i = 2:N, D(i,1) = D(i-1,1) + C_dtw(i,1) + params.skip_penalty; Prev(i,1) = 2; end
        for j = 2:M, D(1,j) = D(1,j-1) + C_dtw(1,j) + params.skip_penalty; Prev(1,j) = 3; end
        
        for i = 2:N
            for j = 2:M
                [v, idx] = min([D(i-1,j-1) + C_dtw(i,j), ...
                               D(i-1,j) + C_dtw(i,j) + params.skip_penalty, ...
                               D(i,j-1) + C_dtw(i,j) + params.skip_penalty]);
                D(i,j) = v; Prev(i,j) = idx;
            end
        end
        
        P = zeros(N, M); ci = N; cj = M;
        while ci >= 1 && cj >= 1
            if mask(ci, cj), P(ci, cj) = 1; end
            if ci == 1 && cj == 1, break; end
            if ci == 1, cj = cj - 1;
            elseif cj == 1, ci = ci - 1;
            else
                step = Prev(ci, cj);
                if step == 1, ci = ci-1; cj = cj-1;
                elseif step == 2, ci = ci-1;
                else, cj = cj-1; end
            end
        end
        P_norm = P / max(sum(P(:)), 1e-10);
        
        mu_s = TS * sum(P_norm, 2);
        mu_d = TD * sum(P_norm, 1)';
        H = (TS - mu_s) * P_norm * (TD - mu_d)';
        [U, ~, V] = svd(H);
        R_new = U * V';
        if det(R_new) < 0, V(:,end) = -V(:,end); R_new = U * V'; end
        t_new = mu_s - R_new * mu_d;
        
        R = R_new; t = t_new;
        curr_cost = sum(sum(P_norm .* C));
        if abs(last_cost - curr_cost) < params.tol, break; end
        last_cost = curr_cost;
    end
    results(m).cost = curr_cost;
    results(m).R = R; results(m).t = t;
end

%% 4. Results Reporting
[~, best_id] = min([results.cost]);
fprintf('\n================ Identification Report ================\n');
fprintf('Candidate 1 (Teammate) Total Cost: %.4f\n', results(1).cost);
fprintf('Candidate 2 (Distractor) Total Cost: %.4f\n', results(2).cost);
fprintf('>>> Result: Identified Candidate %d as Teammate <<<\n', best_id);

%% 5. LaTeX 双栏标准尺寸绘图 + 导出PDF
TD_aligned = results(best_id).R * candidates{best_id} + results(best_id).t;

% LaTeX双栏标准尺寸：宽3.5in 高3.0in
fig_width = 3.5;   
fig_height = 3.0;  

% ========== 图1：原始轨迹 Fig1_Raw.pdf ==========
figure('Units','inches','Position',[1,1,fig_width,fig_height],'Color','w');
hold on; grid on; box on;

p1 = plot(TS(1,:), TS(2,:), 'b-o', 'MarkerSize',4,'MarkerFaceColor','b');
p2 = plot(TD1(1,:), TD1(2,:), 'g-s', 'MarkerSize',4,'MarkerFaceColor','g');
p3 = plot(TD2(1,:), TD2(2,:), 'r-x', 'MarkerSize',4);

xlabel('X (m)');
ylabel('Y (m)');
title('Raw Trajectory Observations');
axis equal;

% 英文学术精简图例（严格对应你的物理含义）
legend([p1,p2,p3],{...
    'Self-estimated trajectory',...
    'LiDAR observed trajectory',...
    'Virtual interference trajectory'});

% 导出无白边矢量PDF
set(gcf,'PaperPositionMode','auto');
print(gcf,'Fig1_Raw','-dpdf','-vector');

% ========== 图2：对齐后轨迹 Fig2_Aligned.pdf ==========
figure('Units','inches','Position',[2,1,fig_width,fig_height],'Color','w');
hold on; grid on; box on;

p4 = plot(TS(1,:), TS(2,:), 'b-o', 'MarkerSize',4,'MarkerFaceColor','b');
p5 = plot(TD_aligned(1,:), TD_aligned(2,:), 'm--d', 'MarkerSize',5,'MarkerFaceColor','m');

for k = 1:5:N
    line([TS(1,k), TD_aligned(1,k)], [TS(2,k), TD_aligned(2,k)],...
         'Color',[0.4 0.4 0.4],'LineStyle',':','HandleVisibility','off');
end

xlabel('X (m)');
ylabel('Y (m)');
title('Optimal Trajectory Alignment');
axis equal;
legend([p4,p5],{'Reference trajectory','Aligned trajectory'});

% 导出无白边矢量PDF
set(gcf,'PaperPositionMode','auto');
print(gcf,'Fig2_Aligned','-dpdf','-vector');

%% 6. Pose Estimation Analysis
fprintf('\n================ Pose Estimation Summary (Candidate %d) ================\n', best_id);
theta_est_rad = atan2(results(best_id).R(2,1), results(best_id).R(1,1));
theta_est_deg = -rad2deg(theta_est_rad);
t_est = results(best_id).t;

fprintf('Estimated Rotation Angle: %.2f deg\n', theta_est_deg);
fprintf('Estimated Translation:    [X: %.4f, Y: %.4f]\n', t_est(1), t_est(2));

if best_id == 1
    fprintf('-------------------------------------------------------\n');
    fprintf('Ground Truth Comparison:\n');
    fprintf('Rotation Error:    %.4f deg\n', abs(theta_est_deg - rad2deg(theta_true)));
    fprintf('Translation Error: %.4f m (Euclidean)\n', norm(t_est - t_true));
end