%% reproduce_fig3_paper_style.m  (Paper-faithful)
% Deep learning based partial-pilot channel estimation (Lee & Sim, 2024)
clear; close all; clc;
rng(0,'twister');

%% ---------------- System / Paper params ----------------
NT   = 64;         % Tx antennas
NR   = 2;           % Rx antennas
NT_H = 8; NT_V = 8;

% 안테나 수가 맞지 않을 때 오류 메시지를 출력함
% 해당 메시지에서 NT 값에 대해 NT_H 와 NT_V 의 곱이 일치하도록 변수 수정
assert(NT_H*NT_V==NT,'UPA size mismatch');   

% 인접한 송신(또는 배열) 안테나 간의 공간적 상관 계수
rho  = 0.9;         % nearest-neighbor corr.
% r_list = [0.875 0.938 0.969];
r_list = [0.5 0.75 0.875];
% Npilot_dict = containers.Map([0.875 0.938 0.969],[56 60 62]);  % |P|
% Npilot_dict = containers.Map([0.5 0.75 0.9],[64 96 115]);
Npilot_dict = containers.Map([0.5 0.75 0.875],[32 48 56]);

% Pilot repetition (NL times per pilot group)
N_L_train = 8; 
N_L_test  = 8;      % same as paper setting idea

% SNR settings (QPSK -> M=4, log(M)=2)
% EbN0_dBs = 0:5:20; 
% mod_order = 4;
bits_per_symbol = 2;

mod_order = 2;                 % QPSK
k = log2(mod_order);           % = 2
EbN0_dBs = 0:5:20;
EsN0_dBs = EbN0_dBs + 10*log10(k);

% EsN0_dBs = EbN0_dBs + 10*log10(bits_per_symbol);
EbN0_train_dB = 20;
% EsN0_train_dB = EbN0_train_dB + 10*log10(bits_per_symbol);
EsN0_train_dB = EbN0_train_dB + 10*log10(k);

% Monte Carlo
N_train = 2000; 
N_test  = 1000; 
valFraction = 0.15;

%% ---------------- Tx correlation Rt (Kronecker) ----------------
R_h = toeplitz(rho.^(0:NT_H-1));
R_v = toeplitz(rho.^(0:NT_V-1));
R_t = kron(R_h, R_v); R_t = (R_t+R_t')/2;       % Hermitianize

% stable Hermitian sqrt
[U,D]=eig(R_t);
lam = max(real(diag(D)),0);
Rts = U*diag(sqrt(lam))*U';

%% ---------------- Channel generator -------------------
gen_channel = @(ns) apply_corr(gen_channel_peda(ns, NR, NT), Rts); % H=Hw*Rt^1/2

%% ---------------- Estimators ---------------------------
% Full LMMSE on a full-length observation vector y (per-Rx row)
lmmse_full  = @(y,s2)  R_t*((R_t + s2*eye(NT))\y);          % conventional
% From partial pilots (indices P) to full vector (paper Eq. 14~15 context)
lmmse_from_partial = @(yP,P,s2) lmmse_from_partial_fn(yP,P,s2,R_t);

%% ---------------- Baseline (Conventional) NMSE ----------------
% Full pilots for all NT, LMMSE, closed-form error covariance Sigma_e
nmse_lmmse_cf = @(Rt,s2,NL) real(trace(inv(inv(Rt)+(NL/s2)*eye(size(Rt))))/trace(Rt));
numSNR = numel(EbN0_dBs);
% results_baseline = zeros(numSNR,1);
% for si=1:numSNR
%     Es_lin = 10.^(EsN0_dBs(si)/10);
%     s2 = 1/Es_lin;
%     results_baseline(si) = nmse_lmmse_cf(R_t, s2, N_L_test);
% end

% === Baseline by simulation (full pilots, NL averaging) ===
results_baseline = zeros(numel(EbN0_dBs),1);

% for si = 1:numel(EbN0_dBs)
%     EsN0_lin   = 10^(EsN0_dBs(si)/10);   % <-- EbN0가 아니라 EsN0 사용
% 
% 
% % for si = 1:numel(EbN0_dBs)
% %     EbN0_lin   = 10^(EbN0_dBs(si)/10);
% %     EsN0_lin   = bits_per_symbol * EbN0_lin;      % ★
%     sigma2     = 1/EsN0_lin;
%     sigma2_eff = sigma2 / N_L_test;               % ★


for si = 1:numel(EbN0_dBs)
    EsN0_lin = 10^(EsN0_dBs(si)/10); % 여기서는 k를 다시 곱하지 않음
    sigma2   = 1/EsN0_lin;
    sigma2_eff = sigma2 / N_L_test;


    tot = 0;
    for t = 1:N_test
        H = gen_channel(1); H = squeeze(H);       % (NR x NT)

        H_est = zeros(NR,NT);
        for rcv = 1:NR
            y = zeros(1,NT);
            for l = 1:N_L_test
                noise = (randn(1,NT)+1i*randn(1,NT)) * sqrt(sigma2/2);
                y = y + (H(rcv,:) + noise);       % 파일럿 심볼은 1로 가정
            end
            y = y / N_L_test;
            h_hat = R_t * ((R_t + sigma2_eff*eye(NT)) \ y.'); % LMMSE
            H_est(rcv,:) = h_hat.';
        end

        e   = norm(H - H_est,'fro')^2;
        pwr = norm(H,'fro')^2;
        tot = tot + e/(pwr + 1e-12);
    end
    results_baseline(si) = tot / N_test;
end

%% ---------------- DNN hyperparams ---------------------
hidden_units  = [1024 1024 1024];  % paper-scale width
learning_rate = 1e-5;              % slightly larger for faster convergence
maxEpochs     = 50;               % paper-like long training
miniBatchSize = 32;

%% ---------------- Main loop over r --------------------
results_dnn_map = containers.Map('KeyType','double','ValueType','any');

for rr=1:numel(r_list)
    r = r_list(rr); 
    Npilot = Npilot_dict(r);
    % 기존: [P_idx, N_idx] = make_uniform_pilots(NT_H, NT_V, Npilot);
    [P_idx, N_idx] = make_uniform_nulls_spread(NT_H, NT_V, Npilot);

    input_dim  = 2*NR*Npilot;
    output_dim = 2*NR*(NT-Npilot);
    fprintf('\n=== r=%.3f | Npilot=%d ===\n',r,Npilot);

    %% --------- Build TRAIN set: input = Hhat_pilot(LMMSE@P), target = Hhat_null(LMMSE@N) ---------
    % Es_lin_train = 10.^(EsN0_train_dB/10); 
    % s2_train = 1/Es_lin_train;

    EbN0_train_dB = 20;
    EsN0_train_dB = EbN0_train_dB + 10*log10(k);
    s2_train = 1/10^(EsN0_train_dB/10);

    X_all = zeros(N_train, input_dim,  'single');
    Y_all = zeros(N_train, output_dim, 'single');

    for n=1:N_train
        H = gen_channel(1); H = squeeze(H);    % (NR x NT), true channel sample

        % Get full-length LMMSE from partial pilots with NL averaging
        H_full_fromP = zeros(NR,NT);
        for rx=1:NR
            yP = zeros(1,length(P_idx));
            for l=1:N_L_train
                noise = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(s2_train/2);
                yP = yP + (H(rx,P_idx) + noise);
            end
            yP = yP / N_L_train;
            h_hat = lmmse_from_partial(yP.', P_idx, s2_train/N_L_train);
            H_full_fromP(rx,:) = h_hat.';
        end

        % Input: LMMSE at pilot indices; Target: LMMSE at null indices (paper training uses Hhat) 
        % (Assuming H_null ≈ Hhat_null at high SNR) 
        X = H_full_fromP(:,P_idx);
        Y = H_full_fromP(:,N_idx);

        x = [real(X(:)); imag(X(:))]';
        y = [real(Y(:)); imag(Y(:))]';
        X_all(n,:) = single(x);
        Y_all(n,:) = single(y);
    end

    % Train/Val split & normalization (X만 정규화; Y는 그대로 사용해도 됨)
    cut = floor((1-valFraction)*N_train);
    Xtr = X_all(1:cut,:);  Ytr = Y_all(1:cut,:);
    Xva = X_all(cut+1:end,:); Yva = Y_all(cut+1:end,:);

    muX = mean(Xtr,1); sX = std(Xtr,0,1)+1e-12;
    XtrN = single((Xtr - muX)./sX);
    XvaN = single((Xva - muX)./sX);

    % DNN (fully-connected with ReLU; MSE loss) — paper's setup 취지 :contentReference[oaicite:5]{index=5}
    layers = [
        featureInputLayer(input_dim,'Normalization','none','Name','in')
        fullyConnectedLayer(hidden_units(1),'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(hidden_units(2),'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(hidden_units(3),'Name','fc3')
        reluLayer('Name','relu3')
        fullyConnectedLayer(output_dim,'Name','out')
        regressionLayer('Name','mse')
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

    %% --------- TEST sweep over SNR: predict H_null and assemble H_bar ---------
    NMSE_dnn = zeros(numSNR,1);
    % for si=1:numSNR
    %     % SNR 설정 -> 잡음 분산 계산
    %     Es_lin = 10^(EsN0_dBs(si)/10); s2 = 1/Es_lin;
    for si=1:numSNR
        Es_lin_train = 10^(EsN0_train_dB/10);  s2_train = 1/Es_lin_train;
        Es_lin = 10^(EsN0_dBs(si)/10);  % <-- EsN0
        s2 = 1/Es_lin;


        acc = 0;
        for t=1:N_test
        % 몬테카를로 반복(N_test) - 매 반복에서 "진짜 채널" H를 생성
            H = gen_channel(1); H = squeeze(H);   % (NR x NT)

            % partial pilots → LMMSE(full) with NL averaging
            % 부분 파일럿 관측 -> LMMSE로 Full 길이 벡터 복원
            % 각 수신 안테나마다 파일럿 위치만 관측(파일럿 심볼은 1 가정)
            % N_L_test 번 평균한 뒤 psrtial -> full LMMSE 적용

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
            % 결과적으로 H_full_fromP 는 전체 송신 안테나 방향의 LMMSE 추정값 (pilot&null 모두)

            % DNN 입력 만들기 → 예측
            % DNN은 파일럿 위치의 LMMSE 추정값만느올 입력을 받고 Null 위치의 채널을 직접 예측
            % 실수부-허수부 분리-결합
            Xin  = H_full_fromP(:,P_idx);
            xin  = [real(Xin(:)); imag(Xin(:))]';
            xinN = single((xin - muX)./sX);
            yhat = predict(net, xinN);

            % 복소 복원
            half = numel(yhat)/2;
            yR = yhat(1:half); yI = yhat(half+1:end);
            Hnull_hat = reshape(yR+1i*yI, NR, numel(N_idx));

            % 최종 조립: P는 LMMSE, N은 DNN
            % 파일럿 구간은 LMMSE 구간을 사용(H_full_fromP(:,P_idx))
            % Null 구간은 DNN 예측 값 (Hnull_hat) 사용
            H_bar = zeros(NR,NT);
            H_bar(:,P_idx) = H_full_fromP(:,P_idx);
            H_bar(:,N_idx) = Hnull_hat;

            % NMSE
            % 각 SNR마다 평균 NMSE를 얻고, results_dnn_map(r)에 저장.
            % 마지막에 네트워크와 정규화 통계, 인덱스들을 .mat로 저장.
            acc = acc + norm(H - H_bar,'fro')^2/(norm(H,'fro')^2 + 1e-12);
        end
        NMSE_dnn(si) = acc / N_test;
        fprintf('r=%.3f  Eb/N0=%2d dB  NMSE=%.4e\n', r, EbN0_dBs(si), NMSE_dnn(si));
    end

    results_dnn_map(r) = NMSE_dnn;
    save(sprintf('net_paperstyle_r%03d.mat',round(1000*r)),'net','muX','sX','P_idx','N_idx');
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

% function [P_idx, N_idx] = make_uniform_pilots(NH, NV, Npilot)
%     % Uniformly spread pilot antennas on UPA grid
%     Hs = unique(round(linspace(1,NH,min(NH,Npilot))),'stable');
%     Vs = unique(round(linspace(1,NV,ceil(Npilot/numel(Hs)))),'stable');
%     [HH,VV] = meshgrid(Hs,Vs);
%     P = reshape((VV-1)*NH + HH,1,[]);
%     if numel(P) >= Npilot
%         pick = round(linspace(1,numel(P),Npilot));
%         P_idx = sort(P(pick));
%     else
%         need = Npilot - numel(P);
%         pool = setdiff(1:(NH*NV), P);
%         P_idx = sort([P, pool(round(linspace(1,numel(pool),need)))]);
%     end
%     N_idx = setdiff(1:(NH*NV), P_idx);
% end

function [P_idx, N_idx] = make_uniform_nulls_spread(NH, NV, Npilot)
    NT = NH*NV;
    Nnull = NT - Npilot;

    % 널을 균일 격자로 먼저 뽑음 (가로 nH, 세로 nV)
    nH = max(1, floor(sqrt(Nnull)));
    nV = ceil(Nnull / nH);
    nH = min(nH, NH); nV = min(nV, NV);

    hs = unique(round(linspace(1, NH, nH)),'stable');
    vs = unique(round(linspace(1, NV, nV)),'stable');
    [HH, VV] = meshgrid(hs, vs);
    Ncand = reshape((VV-1)*NH + HH, 1, []);

    % 후보가 많으면 균등 샘플로 Nnull개만 선택
    if numel(Ncand) >= Nnull
        pick = round(linspace(1, numel(Ncand), Nnull));
        N_idx = sort(Ncand(pick));
    else
        need = Nnull - numel(Ncand);
        pool = setdiff(1:NT, Ncand);
        N_idx = sort([Ncand, pool(round(linspace(1,numel(pool),need)))]);
    end

    P_idx = setdiff(1:NT, N_idx);
end
