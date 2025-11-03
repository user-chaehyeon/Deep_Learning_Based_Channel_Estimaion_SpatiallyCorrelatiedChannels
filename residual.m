%% reproduce_fig3_paper_style_residual.m
% Deep learning based partial-pilot channel estimation (Lee & Sim, 2024)
clear; close all; clc;
rng(0,'twister');

%% ---------------- System / Paper params ----------------
NT   = 64;           % Tx antennas
NR   = 2;            % Rx antennas
NT_H = 8; NT_V = 8;
assert(NT_H*NT_V==NT,'UPA size mismatch');

rho  = 0.9;          % nearest-neighbor correlation between adjacent Tx
r_list = [0.5 0.75 0.875];
Npilot_dict = containers.Map([0.5 0.75 0.875],[32 48 56]);   % |P|

% Pilot repetition (NL times per pilot group)
N_L_train = 16;      % 학습은 더 깨끗하게(안정적 학습)
N_L_test  = 8;       % 테스트는 논문과 유사

% Modulation & SNR (방식 A: Eb/N0는 그대로, 선형에서 k만 곱함)
mod_order = 4;                         % QPSK
bits_per_symbol = log2(mod_order);     % k = 2
EbN0_dBs = 0:5:20;

% Monte Carlo
N_train = 5000; 
N_test  = 1000; 
valFraction = 0.15;

%% ---------------- Tx correlation Rt (Kronecker) ----------------
R_h = toeplitz(rho.^(0:NT_H-1));
R_v = toeplitz(rho.^(0:NT_V-1));
R_t = kron(R_h, R_v); R_t = (R_t+R_t')/2;   % Hermitianize

% stable Hermitian sqrt
[U,D]=eig(R_t);
lam = max(real(diag(D)),0);
Rts = U*diag(sqrt(lam))*U';

%% ---------------- Channel generator -------------------
gen_channel = @(ns) apply_corr(gen_channel_peda(ns, NR, NT), Rts); % H=Hw*Rt^{1/2}

%% ---------------- Estimators ---------------------------
% Full LMMSE on a full-length observation vector y (per-Rx row)
lmmse_full  = @(y,s2)  R_t*((R_t + s2*eye(NT))\y);           % conventional
% From partial pilots (indices P) to full vector
lmmse_from_partial = @(yP,P,s2) lmmse_from_partial_fn(yP,P,s2,R_t);

%% ---------------- Baseline (Conventional) NMSE ----------------
% === Baseline by simulation (full pilots, NL averaging) ===
results_baseline = zeros(numel(EbN0_dBs),1);
for si = 1:numel(EbN0_dBs)
    EbN0_lin   = 10^(EbN0_dBs(si)/10);
    SNR_es     = bits_per_symbol * EbN0_lin;   % Es/N0 = k * Eb/N0
    sigma2     = 1/SNR_es;
    sigma2_eff = sigma2 / N_L_test;

    tot = 0;
    for t = 1:N_test
        H = gen_channel(1); H = squeeze(H);       % (NR x NT)
        H_est = zeros(NR,NT);
        for rcv = 1:NR
            y = zeros(1,NT);
            for l = 1:N_L_test
                noise = (randn(1,NT)+1i*randn(1,NT)) * sqrt(sigma2/2);
                y = y + (H(rcv,:) + noise);       % pilot symbol power = 1
            end
            y = y / N_L_test;
            h_hat = R_t * ((R_t + sigma2_eff*eye(NT)) \ y.'); % LMMSE
            H_est(rcv,:) = h_hat.';
        end
        tot = tot + norm(H - H_est,'fro')^2/(norm(H,'fro')^2 + 1e-12);
    end
    results_baseline(si) = tot / N_test;
end

%% ---------------- DNN hyperparams ---------------------
hidden_units  = [1024 1024 1024 1024];  % paper-scale width
learning_rate = 1e-5;
maxEpochs     = 50;                % residual 학습이라 30~60 무난
miniBatchSize = 32;

%% ---------------- Main loop over r --------------------
results_dnn_map = containers.Map('KeyType','double','ValueType','any');

for rr=1:numel(r_list)
    r = r_list(rr); 
    Npilot = Npilot_dict(r);
    % 파일럿을 균일하게 퍼뜨려 커버리지 확보 (r ↓일수록 중요)
    [P_idx, N_idx] = make_uniform_pilots(NT_H, NT_V, Npilot);

    input_dim  = 2*NR*Npilot + 1;             % +1: SNR feature
    output_dim = 2*NR*(NT-Npilot);
    fprintf('\n=== r=%.3f | Npilot=%d ===\n',r,Npilot);

    %% --------- Build TRAIN set (Residual target, Multi-SNR) ---------
    X_all = zeros(N_train, input_dim,  'single');
    Y_all = zeros(N_train, output_dim, 'single');

    for n=1:N_train
        % 1) True channel (NR x NT)
        H = gen_channel(1); H = squeeze(H);

        % 2) 샘플별 스케일(정규화용)
        scale = sqrt(mean(abs(H(:)).^2) + 1e-12);

        % 3) 학습 SNR 랜덤 선택 (멀티-SNR)
        EbN0_dB_cur = EbN0_dBs(randi(numel(EbN0_dBs)));
        EbN0_lin_cur = 10^(EbN0_dB_cur/10);
        SNR_es_cur   = bits_per_symbol * EbN0_lin_cur;  % Es/N0
        s2_train_cur = 1/SNR_es_cur;                    % noise var
        snr_feat     = single(sqrt(SNR_es_cur));        % DNN 입력 feature

        % 4) partial pilots → LMMSE(full) with NL averaging
        H_full_fromP = zeros(NR,NT);
        for rx=1:NR
            yP = zeros(1,length(P_idx));
            for l=1:N_L_train
                noise = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(s2_train_cur/2);
                yP = yP + (H(rx,P_idx) + noise);
            end
            yP = yP / N_L_train;
            h_hat = lmmse_from_partial(yP.', P_idx, s2_train_cur/N_L_train);
            H_full_fromP(rx,:) = h_hat.';
        end

        % 5) 입력: 파일럿 위치의 LMMSE (per-sample scale로 정규화)
        Xin  = H_full_fromP(:,P_idx)/scale;

        % 6) 타깃: Residual = (정답 − LMMSE) at nulls (동일 스케일)
        Yres = (H(:,N_idx) - H_full_fromP(:,N_idx))/scale;

        % 7) 실/허 분리 + SNR feature 결합
        x = [real(Xin(:)); imag(Xin(:)); snr_feat]';
        y = [real(Yres(:)); imag(Yres(:))]';

        X_all(n,:) = single(x);
        Y_all(n,:) = single(y);
    end

    % Train/Val split & 통계 정규화(입력만)
    cut = floor((1-valFraction)*N_train);
    Xtr = X_all(1:cut,:);  Ytr = Y_all(1:cut,:);
    Xva = X_all(cut+1:end,:); Yva = Y_all(cut+1:end,:);

    muX = mean(Xtr,1); sX = std(Xtr,0,1)+1e-12;
    XtrN = single((Xtr - muX)./sX);
    XvaN = single((Xva - muX)./sX);

    % DNN (Residual 회귀)
    layers = [
        featureInputLayer(input_dim,'Normalization','none','Name','in')
        fullyConnectedLayer(hidden_units(1),'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(hidden_units(2),'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(hidden_units(3),'Name','fc3')
        reluLayer('Name','relu3')
        fullyConnectedLayer(output_dim,'Name','out')
        regressionLayer('Name','mse')   % per-sample scale 덕에 MSE로도 NMSE에 근접
    ];
    options = trainingOptions('adam', ...
        'InitialLearnRate',learning_rate, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{XvaN,Yva}, ...
        'ValidationPatience',20, ...
        'Verbose',true, ...
        'Plots','training-progress', ...
        'ExecutionEnvironment','auto');

    net = trainNetwork(XtrN, Ytr, layers, options);

    %% --------- TEST sweep over SNR: predict residual@null & assemble ---------
    NMSE_dnn = zeros(numel(EbN0_dBs),1);

    for si=1:numel(EbN0_dBs)
        EbN0_lin = 10^(EbN0_dBs(si)/10);
        SNR_es   = bits_per_symbol * EbN0_lin;   % Es/N0
        s2       = 1/SNR_es;
        snr_feat = single(sqrt(SNR_es));
        acc = 0;

        for t=1:N_test
            H = gen_channel(1); H = squeeze(H);   % (NR x NT)

            % per-sample scale (테스트에서도 동일하게)
            scale = sqrt(mean(abs(H(:)).^2) + 1e-12);

            % partial → LMMSE(full) with NL averaging
            H_full_fromP = zeros(NR,NT);
            for rx=1:NR
                yP = zeros(1,length(P_idx));
                for l=1:N_L_test
                    noise = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(s2/2);
                    yP = yP + (H(rx,P_idx) + noise);
                end
                yP = yP / N_L_test;
                h_hat = lmmse_from_partial(yP.', P_idx, s2/N_L_test);
                H_full_fromP(rx,:) = h_hat.';
            end

            % DNN 입력(정규화+SNR feature)
            Xin  = H_full_fromP(:,P_idx)/scale;
            xin  = [real(Xin(:)); imag(Xin(:)); snr_feat]';
            xinN = single((xin - muX)./sX);

            % Residual 예측
            yhat = predict(net, xinN);
            half = numel(yhat)/2;
            yR = yhat(1:half); yI = yhat(half+1:end);
            Res_hat = reshape(yR+1i*yI, NR, numel(N_idx));  % (정규화된 공간)

            % 스케일 복원 + LMMSE에 residual 보정
            H_bar = zeros(NR,NT);
            H_bar(:,P_idx) = H_full_fromP(:,P_idx);
            H_bar(:,N_idx) = H_full_fromP(:,N_idx) + Res_hat*scale;

            % NMSE
            acc = acc + norm(H - H_bar,'fro')^2/(norm(H,'fro')^2 + 1e-12);
        end
        NMSE_dnn(si) = acc / N_test;
        fprintf('r=%.3f  Eb/N0=%2d dB  NMSE=%.4e\n', r, EbN0_dBs(si), NMSE_dnn(si));
    end

    results_dnn_map(r) = NMSE_dnn;
    save(sprintf('net_residual_r%03d.mat',round(1000*r)),'net','muX','sX','P_idx','N_idx');
end

%% ---------------- Plot (match paper style) ----------------
figure('Color','w','Position',[100 100 900 600]);
semilogy(EbN0_dBs, results_baseline, 'b-','LineWidth',1.5); hold on;
markers = {'r^-','r>-','rv-'};
for i=1:numel(r_list)
    r = r_list(i);
    semilogy(EbN0_dBs, results_dnn_map(r), markers{i}, 'LineWidth',1.2, 'MarkerSize',6);
end
grid on; box on;
xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('NMSE','FontSize',16);
title(sprintf('NMSE vs E_b/N_0 (N_T=%d, N_R=%d)',NT,NR),'FontSize',16);
ylim([1e-4 1e-1]); yticks([1e-4 1e-3 1e-2 1e-1]);
legend({'Conventional method', ...
        sprintf('Proposed method (r = %.3f)',r_list(1)), ...
        sprintf('Proposed method (r = %.3f)',r_list(2)), ...
        sprintf('Proposed method (r = %.3f)',r_list(3))}, ...
        'Location','southwest','FontSize',12);

%% ================= Local functions =================
function Hc = apply_corr(H, Rts)
    [ns, NR, NT] = size(H);
    Hc = zeros(ns,NR,NT);
    for k=1:ns
        Hw = squeeze(H(k,:,:));      % NR x NT
        Hc(k,:,:) = Hw * Rts;        % TX-side correlation
    end
end

function H = gen_channel_peda(ns, NR, NT)
    % Flat-fading sample using PedA PDP weights (no Doppler)
    p_db  = [0 -1 -9 -10 -15 -20];
    p_lin = 10.^(p_db/10); p_lin = p_lin/sum(p_lin);
    H = zeros(ns,NR,NT);
    for k=1:ns
        for rx=1:NR
            taps = (randn(NT,length(p_lin))+1i*randn(NT,length(p_lin))).*sqrt(p_lin/2);
            hrow = sum(taps,2).';                 % 1 x NT
            H(k,rx,:) = hrow;
        end
    end
end

function h_hat = lmmse_from_partial_fn(yP, P, s2, Rt)
    % yP: Mx1; P: 1xM indices
    P = P(:).';
    Rpp = Rt(P,P); Rtp = Rt(:,P);
    h_hat = Rtp * ((Rpp + s2*eye(numel(P)))\yP);
end

function [P_idx, N_idx] = make_uniform_pilots(NH, NV, Npilot)
    % Uniformly spread pilot antennas on UPA grid (파일럿 커버리지 보장)
    NT = NH*NV;
    Hs = unique(round(linspace(1,NH,min(NH,Npilot))),'stable');
    Vs = unique(round(linspace(1,NV,ceil(Npilot/numel(Hs)))),'stable');
    [HH,VV] = meshgrid(Hs,Vs);
    P = reshape((VV-1)*NH + HH,1,[]);
    if numel(P) >= Npilot
        pick = round(linspace(1,numel(P),Npilot));
        P_idx = sort(P(pick));
    else
        need = Npilot - numel(P);
        pool = setdiff(1:(NH*NV), P);
        P_idx = sort([P, pool(round(linspace(1,numel(pool),need)))]);
    end
    N_idx = setdiff(1:(NH*NV), P_idx);
end
