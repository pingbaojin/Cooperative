clc; clear; close all;

% ========== 参数设置 ==========
s_th = 0.5;              
max_missed = 5;          
dt = 0.1;                
num_frames = 60;       
fit_win = 10;            
poly_order = 2;          

% 卡尔曼合理参数
Q = 0.1 * eye(6);  
R = 0.2 * eye(3);  

% 初始化
tracks = {};
next_id = 1;
true_all   = cell(1, num_frames);
ekf_all    = cell(1, num_frames);
fit_all    = cell(1, num_frames);

% ===================== 主循环 =====================
for t = 1:num_frames
    % 1. 生成：运动全程平稳 + 分段噪声（仅观测误差变化）
    [O_t, true_pos] = simulateDetections(t);
    true_all{t} = true_pos;

    % 2. 置信度筛选
    valid_idx = O_t(:,4) > s_th;
    detections = O_t(valid_idx, 1:3);

    % 3. 数据关联
    [assignments, unassigned_tracks, unassigned_dets] ...
        = associateTracks(tracks, detections);

    % 4. 卡尔曼更新 + 缓存观测
    for i = 1:size(assignments,1)
        trk_idx = assignments(i,1);
        det_idx = assignments(i,2);
        z = detections(det_idx,:)';

        tracks{trk_idx} = kalmanUpdate(tracks{trk_idx}, z, dt, Q, R);
        tracks{trk_idx}.missed = 0;

        tracks{trk_idx}.det_hist = [tracks{trk_idx}.det_hist, z];
        if size(tracks{trk_idx}.det_hist,2) > fit_win
            tracks{trk_idx}.det_hist = tracks{trk_idx}.det_hist(:,end-fit_win+1:end);
        end
    end

    % 5. 未匹配轨迹丢失计数
    for i = 1:length(unassigned_tracks)
        idx = unassigned_tracks(i);
        tracks{idx}.missed = tracks{idx}.missed + 1;
    end

    % 6. 删除过期轨迹
    tracks = tracks(cellfun(@(x) x.missed <= max_missed, tracks));

    % 7. 新建轨迹
    for i = 1:length(unassigned_dets)
        det_idx = unassigned_dets(i);
        pos0 = detections(det_idx,:)';
        new_trk = initTrack(pos0, next_id);
        new_trk.det_hist = pos0;
        tracks{end+1} = new_trk;
        next_id = next_id + 1;
    end

    % 计算卡尔曼 + 多项式拟合
    curr_ekf = [];
    curr_fit = [];
    for i = 1:length(tracks)
        tid    = tracks{i}.id;
        ekf_pos= tracks{i}.x(1:3)';
        hist_p = tracks{i}.det_hist';
        
        if size(hist_p,1) >= poly_order+1
            fit_pos = polyFitTraj(hist_p, poly_order);
        else
            fit_pos = ekf_pos;
        end

        curr_ekf = [curr_ekf; ekf_pos, tid];
        curr_fit = [curr_fit; fit_pos, tid];
    end
    ekf_all{t} = curr_ekf;
    fit_all{t} = curr_fit;

    % 实时可视化
    visualizeCompare(tracks, true_pos, curr_fit, t, num_frames);
end

% 误差统计绘图
calcErrorCompare(true_all, ekf_all, fit_all, num_frames);

% =========================================================================
% 子函数
% =========================================================================
function track = initTrack(pos, id)
    track.id       = id;
    track.x        = [pos; 0; 0; 0];
    track.P        = eye(6);
    track.missed   = 0;
    track.det_hist = [];
end

function track = kalmanUpdate(track, z, dt, Q, R)
    F = [eye(3), dt*eye(3);
         zeros(3), eye(3)];
    H = [eye(3), zeros(3)];

    x_pred = F * track.x;
    P_pred = F * track.P * F' + Q;

    K = P_pred * H' / (H * P_pred * H' + R);
    x_upd = x_pred + K * (z - H * x_pred);
    P_upd = (eye(6) - K * H) * P_pred;

    track.x = x_upd;
    track.P = P_upd;
end

function [assignments, unassigned_tracks, unassigned_dets] ...
    = associateTracks(tracks, detections)
    nT = length(tracks);
    nD = size(detections,1);
    if nT==0 || nD==0
        assignments = zeros(0,2);
        unassigned_tracks = 1:nT;
        unassigned_dets = 1:nD;
        return;
    end
    cost = zeros(nT,nD);
    for i=1:nT
        pred_p = tracks{i}.x(1:3);
        for j=1:nD
            cost(i,j) = norm(pred_p - detections(j,:)');
        end
    end
    assignments = []; used_dets = [];
    for i=1:nT
        [min_val,idx] = min(cost(i,:));
        if min_val < 2.5
            assignments = [assignments; i,idx];
            used_dets = [used_dets,idx];
        end
    end
    unassigned_tracks = setdiff(1:nT, assignments(:,1));
    unassigned_dets   = setdiff(1:nD, used_dets);
end

% 核心修改：运动全程恒定不变，只分段改变观测噪声
function [O_t, true_pos] = simulateDetections(t)
    % ========== 目标运动：全程平稳，无任何突变 ==========
    p1 = [t*0.1, sin(t*0.1), 0];
    p2 = [5 + cos(t*0.1), t*0.05, 0];
    true_pos = [p1; p2];

    % ========== 仅观测误差分段变化 ==========
    if t <= 30
        noise_scale = 0.05;   % 前30帧：小观测噪声
    else
        noise_scale = 0.5;   % 后30帧：观测误差大幅变大
    end

    noise = noise_scale * randn(2,3);  
    O_t = [p1+noise(1,:), 0.9; p2+noise(2,:), 0.8];
end

% 多项式最小二乘拟合
function fit_pos = polyFitTraj(pts, order)
    fit_pos = zeros(1,3);
    t_seq = 1:size(pts,1);
    for dim = 1:3
        coeff = polyfit(t_seq, pts(:,dim), order);
        fit_pos(dim) = polyval(coeff, t_seq(end));
    end
end

% 实时可视化
function visualizeCompare(tracks, true_pos, fit_pts, frame, total)
    clf; hold on; grid on;
    plot3(true_pos(:,1), true_pos(:,2), true_pos(:,3),'g+','MarkerSize',12,'LineWidth',2);

    % 卡尔曼：红圈
    for i = 1:length(tracks)
        p = tracks{i}.x(1:3);
        plot3(p(1),p(2),p(3),'ro','MarkerSize',6,'LineWidth',1.2);
    end

    % 多项式拟合：品红方形
    for i = 1:size(fit_pts,1)
        p = fit_pts(i,1:3);
        plot3(p(1),p(2),p(3),'ms','MarkerSize',6,'LineWidth',1.2);
    end

    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('帧%d/%d  绿=真实 | 红=卡尔曼 | 品红=多项式拟合',frame,total));
    legend('真实位置','卡尔曼滤波','多项式拟合','Location','best');
    axis equal; xlim([0,8]); ylim([-2,2]); zlim([-0.1,0.1]);
    drawnow; pause(0.03);
end

% 误差计算与绘图
function calcErrorCompare(true_all, ekf_all, fit_all, N)
    err_ekf_1 = zeros(1,N); err_fit_1 = zeros(1,N);
    err_ekf_2 = zeros(1,N); err_fit_2 = zeros(1,N);

    for t = 1:N
        if isempty(true_all{t}), continue; end
        t1 = true_all{t}(1,:);
        t2 = true_all{t}(2,:);

        if ~isempty(ekf_all{t})
            ekf1 = ekf_all{t}(ekf_all{t}(:,4)==1, 1:3);
            ekf2 = ekf_all{t}(ekf_all{t}(:,4)==2, 1:3);
            if ~isempty(ekf1), err_ekf_1(t) = norm(ekf1 - t1); end
            if ~isempty(ekf2), err_ekf_2(t) = norm(ekf2 - t2); end
        end
        if ~isempty(fit_all{t})
            fit1 = fit_all{t}(fit_all{t}(:,4)==1, 1:3);
            fit2 = fit_all{t}(fit_all{t}(:,4)==2, 1:3);
            if ~isempty(fit1), err_fit_1(t) = norm(fit1 - t1); end
            if ~isempty(fit2), err_fit_2(t) = norm(fit2 - t2); end
        end
    end

    % 指标
    mean_ekf1 = mean(err_ekf_1(err_ekf_1>0));
    mean_fit1  = mean(err_fit_1(err_fit_1>0));
    rmse_ekf1  = sqrt(mean(err_ekf_1(err_ekf_1>0).^2));
    rmse_fit1  = sqrt(mean(err_fit_1(err_fit_1>0).^2));

    mean_ekf2 = mean(err_ekf_2(err_ekf_2>0));
    mean_fit2  = mean(err_fit_2(err_fit_2>0));
    rmse_ekf2  = sqrt(mean(err_ekf_2(err_ekf_2>0).^2));
    rmse_fit2  = sqrt(mean(err_fit_2(err_fit_2>0).^2));

    fprintf('\n========== 全局误差对比 ==========\n');
    fprintf('【目标1】\n');
    fprintf('卡尔曼｜平均误差:%.4f  RMSE:%.4f\n',mean_ekf1,rmse_ekf1);
    fprintf('多项式拟合｜平均误差:%.4f  RMSE:%.4f\n',mean_fit1,rmse_fit1);
    fprintf('【目标2】\n');
    fprintf('卡尔曼｜平均误差:%.4f  RMSE:%.4f\n',mean_ekf2,rmse_ekf2);
    fprintf('多项式拟合｜平均误差:%.4f  RMSE:%.4f\n',mean_fit2,rmse_fit2);

    % 误差曲线：标注「观测噪声增大」
    figure('Name','滤波 VS 拟合 ｜ 观测噪声波动对比');
    subplot(2,1,1);hold on;grid on;
    plot(err_ekf_1,'r-','LineWidth',1.5);
    plot(err_fit_1,'m--','LineWidth',1.5);
    xline(30,'k--','观测误差增大');
    title('目标1 逐帧位置误差');
    legend('卡尔曼滤波','多项式拟合','Location','best');

    subplot(2,1,2);hold on;grid on;
    plot(err_ekf_2,'r-','LineWidth',1.5);
    plot(err_fit_2,'m--','LineWidth',1.5);
    xline(30,'k--','观测误差增大');
    title('目标2 逐帧位置误差');
    legend('卡尔曼滤波','多项式拟合','Location','best');
    xlabel('帧号');
end