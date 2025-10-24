%% reproduce_fig3_dnn_lmmse.m
% MATLAB + Deep Learning Toolbox
clear; close all; clc;

rng(0, 'twister');                                                         % 난수 생성기 시드 고정 (재현성 확보)
clearvars; close all; clc;                                                 % 작업공간 정리 및 화면 초기화

%% ------------------ System / Paper parameters ------------------
NT = 64;                                                                   % 송신 안테나 수 NT (transmit antennas)
NR = 2;                                                                    % 수신 안테나 수 NR (receive antennas)
NT_H = 8; NT_V = 8;                                                        % UPA (Uniform Planar Array) 가로/세로 차원 (NT_H x NT_V = NT)


rho = 0.9;                                                                 % 인접 안테나 간 상관계수 rho (correlation coefficient)
r_list = [0.875, 0.938, 0.969];                                            % pilot ratio 리스트 r
% r_list = [0.5,0.75,0.15];

% r에 대응되는 파일럿 개
Npilot_dict = containers.Map([0.875,0.938,0.969],[56,60,62]);              % r -> Npilot 매핑
% Npilot_dict = containers.Map([0.5,0.75,0.15],[32,48,16]);

%% ------------------ DNN hyperpatameters ------------------
hidden_units = [1024, 1024, 1024];                                         % 은닉층 노드 수 (DNN hidden units)
learning_rate = 1e-5;                                                      % Adam 초기 학습률 (learning rate)
maxEpochs = 30;                                                            % Epoch 수 (논문에서 500)
miniBatchSize = 32;                                                        % 미니배치 크기 (mini-batch size)

%% ------------------ 데이터/훈련 설정 ------------------
N_train = 2000;                                                            % 훈련 샘플 수
valFraction = 0.15;                                                        % 검증 데이터 비율 (validation fraction)
N_test = 1000;                                                             % 테스트 샘플 수 (각 SNR에서 Monte-Carlo 샘플 수)

%% ------------------ 파일럿 반복 및 SNR 설정 ------------------
N_L_train = 8;                                                             % pilot repetition 수 (NL for training)
N_L_test = 8;                                                              % pilot repetition 수 (NL for testing)

EbN0_train_dB = 20;                                                        % DNN 학습시 사용할 Eb/N0
% mod_order = 2;                                                           % 변조 차수 (4-QAM) ---- <<변경>>
% % mod_order = 1 로 두면 bits_per_symbol = log2(1) = 0 -> EsN0 = EbN0 +
% % 10*log10(0) 이 정의되지 않아서 오류 발생


% bits_per_symbol = log2(mod_order);                                         % 심볼 당 비트 수 (QPSK -> 2)
% EsN0_train_dB = EbN0_train_dB + 10*log10(bits_per_symbol);                 % Es/N0 변환 (symbol energy to noise)


mod_order = 4; bits_per_symbol = 2;
EbN0_dBs = 0:5:20;                                                         % 결과 플롯용 Eb/N0 sweep (dB)

EsN0_train_dB = EbN0_train_dB + 10*log10(bits_per_symbol);
EsN0_dBs      = EbN0_dBs      + 10*log10(bits_per_symbol);



%EbN0_dBs = 20:2:40;

%% ------------------ Transmit correlation matrix Rt 구성 (UPA, Kronecker) ------------------
% R(N)_{u,v} = rho^{|u-v|} 형태의 Toeplitz 행렬 생성 (ULA 모델에서 인접 안테나 상관 모델링)
% TX UPA의 공간상관을 Kronecker 모델로 반영항 송신측 공분산 행렬 Rt

R_h = toeplitz(rho.^(0:NT_H-1));                                           % horizontal correlation matrix R_h (NT_H x NT_H)
R_v = toeplitz(rho.^(0:NT_V-1));                                           % vertical correlation matrix R_v (NT_V x NT_V)
R_t = kron(R_h, R_v);                                                      % transmit correlation matrix Rt (NT x NT) via Kronecker product
R_t = (R_t + R_t') / 2;                                                    % 수치적 안정화를 위한 대칭화 (numerical symmetrize)

%% ------------------ Helper function handles ------------------


% Hermitian sqrt (안정적인 Hermit 제곱근 만들기 위한 전처리 - 수치안정)
[U,D] = eig((R_t+R_t')/2);                                                 % 수치적으로 정확한 Hermitian화 (대칭화) 후 고유분해
D = max(real(diag(D)), 0);                                                 % 음수 제거(수치오차)
R_t_sqrt = U*diag(sqrt(D))*U';                                             % 고유값의 실수부만 취하고, 음의 값은 0으로 클리핑 -> PSD로 투영


% 상관 채널 생성
gen_channel = @(ns) apply_corr(gen_channel_peda(ns, NR, NT), R_t_sqrt);

function Hc = apply_corr(H, Rts)
    [ns, NR, NT] = size(H); Hc = zeros(ns, NR, NT);
    for k=1:ns
        Hw = squeeze(H(k,:,:));                                             % NR x NT
        Hc(k,:,:) = Hw * Rts;                                               % Rt^{1/2} (Hermitian) 곱
    end
end

% LMMSE
lmmse_full  = @(y,s2)  R_t * ((R_t + s2*eye(NT)) \ y);
% 모든 송신 안테나가 파일럿을 보낸다고 가정햇을 대, 한 수신 안테나에서 본 전체 송신 방향 관측 - 관측 잡음 분산
lmmse_from_partial = @(yP,P,s2) lmmse_from_partial_fn(yP, P, s2, R_t);
% 일부 송신 안테나만 파일럿을 전송 햇을 대, 파일럿을 보낸 M개의 송신 안테나의 관측값, 


%% ------------------ Precompute Baseline (Full LMMSE) NMSE across SNRs ------------------
% Baseline(Conventional): 모든 안테나에 대해 pilot 전송 -> full LMMSE 적용
numSNR = numel(EbN0_dBs);                                                  % SNR 포인트 개수

function nmse = nmse_lmmse_closedform(Rt, sigma2, NL)
    % Rt: NT x NT, sigma2: noise variance per symbol (Es normalized to 1)
    % NL: repetition count
    A = (inv(Rt) + (NL/sigma2)*eye(size(Rt)));
    Sigma_e = inv(A);
    nmse = real(trace(Sigma_e) / trace(Rt));
end

results_baseline = zeros(numSNR,1);                                        % baseline NMSE 결과 벡터 초기화

for si = 1:numSNR
    EsN0_lin = 10^(EsN0_dBs(si)/10);
    sigma2   = 1/EsN0_lin;
    results_baseline(si) = nmse_lmmse_closedform(R_t, sigma2, N_L_test);
end



%% ------------------ Main loop: for each r (train DNN using full pilots then test with partial pilots) ------------------
results_dnn_map = containers.Map('KeyType','double','ValueType','any');    % DNN 결과를 저장할 맵 (r -> NMSE vector)

for rr = 1:numel(r_list)                                                   % r_list 각 원소에 대해 반복
    r = r_list(rr);                                                        % 현재 r 값
    Npilot = Npilot_dict(r);                                               % 해당 r에 대한 파일럿 수 (Table 값)
    P_idx = 1:Npilot;                                                      % 파일럿을 보낸 안테나 인덱스
    % UPA(8×8) 물리 배열을 1차로 펴서 한 족에 몰린 패턴 사용
    N_idx = setdiff(1:NT, P_idx);                                          % 파일럿이 없는 null 인덱스
    input_dim = 2 * NR * Npilot;                                           % DNN 입력 차원 (real+imag)
    output_dim = 2 * NR * (NT - Npilot);                                   % DNN 출력 차원 (real+imag)

    fprintf('\n=== r=%.3f | Npilot=%d | input=%d | output=%d ===\n', r, Npilot, input_dim, output_dim); % 상태 출력

    %% ------------------ Create training dataset (TRAIN with full pilots at EsN0_train_dB) ------------------
    EsN0_lin_train = 10^(EsN0_train_dB/10);                                % 학습용 Es/N0 선형값
    sigma2_train = 1/EsN0_lin_train;                                       % 학습 시 noise variance (symbol-level)
    sigma2_ls_train = sigma2_train / N_L_train;                            % 반복(NL) 고려한 effective noise variance

    X_train_all = zeros(N_train, input_dim, 'single');                     % 훈련 입력 행렬 초기화 (N_train x input_dim)
    Y_train_all = zeros(N_train, output_dim, 'single');                    % 훈련 정답 행렬 초기화 (N_train x output_dim)

    for n = 1:N_train                                                      % 훈련 샘플 생성 루프
        H = gen_channel(1); H = squeeze(H);                                % 실제 채널 H 생성 (NR x NT)
        H_hat_full = zeros(NR, NT);                                        % full LMMSE로 추정된 H 저장용
        for rcv = 1:NR
            noise = (randn(1,NT) + 1i*randn(1,NT)) * sqrt(sigma2_ls_train/2);% AWGN 생성 (학습 상황)
            % y_full = H(rcv,:) + noise;                                     % full observation (행벡터)
            % h_hat = lmmse_full(y_full.', sigma2_ls_train);                 % full LMMSE (열벡터)

            y_full = zeros(1,NT);
            for l = 1:N_L_test
                noise_l = (randn(1,NT) + 1i*randn(1,NT)) * sqrt(sigma2/2); % per-repeat noise: sigma2
                y_full = y_full + (H(rcv,:) + noise_l);
            end
            y_full = y_full / N_L_test;                                    % 평균
            h_hat  = lmmse_full(y_full.', sigma2/N_L_test);                % 등가분산으로 LMMSE

            H_hat_full(rcv,:) = h_hat.';                                   % 추정값을 행으로 저장
        end
        pilot_part = H_hat_full(:, P_idx);                                 % input으로 사용할 pilot 칼럼 (NR x Npilot)
        % null_part = H_hat_full(:, N_idx);                                % output으로 사용할 null 칼럼 (NR x (NT-Npilot))
        null_part = H(:, N_idx);                                           % Y_train_all 생성부
        % 기존에 사용햇던 값은 노이즈 포함한 LMMSE 결과 - 진자 채널 H를 타깃으로 학습 DENOISE+INPAINT 하기 위함 -- 테스트대도 파일럿 위치는 LMMSE/LS 로, NULL 위치만 DNN 예측


        x = [real(pilot_part(:)); imag(pilot_part(:))];                    % input 벡터화: real then imag, column-major
        y = [real(null_part(:)); imag(null_part(:))];                      % output 벡터화: real then imag
        X_train_all(n, :) = single(x.');                                   % 가로에 대해 행 벡터로 저장
        Y_train_all(n, :) = single(y.');                                   % 세로에 대해 행 벡터로 저장
    end

    %% ------------------ Train/Validation split & normalization ------------------
    idx_split = floor((1 - valFraction) * N_train);                           % train/val 분할 인덱스 계산
    XTrain = X_train_all(1:idx_split, :);                                     % 실제 훈련 입력
    YTrain = Y_train_all(1:idx_split, :);                                     % 실제 훈련 타깃

    XVal = X_train_all(idx_split+1:end, :);                                   % 검증 입력
    YVal = Y_train_all(idx_split+1:end, :);                                   % 검증 타깃

    muX = mean(XTrain, 1);                                                    % 입력 평균 (feature-wise)
    sX = std(XTrain, 0, 1) + 1e-12;                                           % 입력 표준편차 (분모 0 방지)
    XTrain_norm = single((XTrain - muX) ./ sX);                               % 훈련 입력 정규화 및 single 변환
    XVal_norm = single((XVal - muX) ./ sX);                                   % 검증 입력 같은 통계로 정규화

    YTrain = single(YTrain);                                                  % Y도 single로 변환 (trainNetwork 요구)
    YVal = single(YVal);                                                      % YVal single 변환

    %% ------------------ Build DNN (Fully-connected regression) ------------------
    layers = [                                                               % DNN 레이어 구성 (fully-connected)
        featureInputLayer(input_dim,'Normalization','none','Name','input')    % 입력 레이어 (feature input)
        fullyConnectedLayer(hidden_units(1),'Name','fc1')                    % 첫 FC 레이어
        reluLayer('Name','relu1')                                             % ReLU 활성화
        fullyConnectedLayer(hidden_units(2),'Name','fc2')                    % 두 번째 FC 레이어
        reluLayer('Name','relu2')                                             % ReLU 활성화
        fullyConnectedLayer(hidden_units(3),'Name','fc3')                    % 세 번째 FC 레이어
        reluLayer('Name','relu3')                                             % ReLU 활성화
        fullyConnectedLayer(output_dim,'Name','fc_out')                       % 출력 FC 레이어 (회귀 출력)
        regressionLayer('Name','reg')                                         % 회귀 손실 레이어 (MSE)
    ];

    options = trainingOptions('adam', ...                                     % 최적화: Adam
        'InitialLearnRate', learning_rate, ...                                % 초기 학습률
        'MaxEpochs', maxEpochs, ...                                          % 최대 epoch
        'MiniBatchSize', miniBatchSize, ...                                  % 배치 크기
        'Shuffle', 'every-epoch', ...                                        % epoch마다 셔플
        'ValidationData', {XVal_norm, YVal}, ...                             % 검증 데이터
        'ValidationFrequency', max(1, floor(idx_split/miniBatchSize)), ...   % 검증 빈도 (0 방지)
        'Verbose', true, ...                                                 % 훈련 출력 활성
        'Plots', 'training-progress', ...                                    % 학습 진행 플롯
        'ExecutionEnvironment', 'auto' );                                    % 실행 환경 자동 선택 (GPU 가능시 GPU 사용)

    % 실제 네트워크 학습 (시간 소요됨)
    net = trainNetwork(XTrain_norm, YTrain, layers, options);                % DNN 학습 수행

    %% ------------------ Evaluate DNN across SNR sweep (NMSE 계산) ------------------
    NMSE_dnn = zeros(numSNR,1);                                             % DNN NMSE 결과 벡터 초기화

    for si = 1:numSNR                                                       % 각 Eb/N0에 대해 테스트
        EbN0_db = EbN0_dBs(si);                                             % 현재 Eb/N0 (dB)
        EsN0_db = EsN0_dBs(si);                                             % 현재 Es/N0 (dB)
        EsN0_lin = 10^(EsN0_db/10);                                         % 선형 Es/N0
        sigma2 = 1/EsN0_lin;                                                % noise variance
        sigma2_ls = sigma2 / N_L_test;                                      % repetition 고려 sigma2

        tot_nmse_dnn = 0;                                                   % DNN 누적 NMSE
        for t = 1:N_test                                                    % 테스트 샘플 루프
            H = gen_channel(1); H = squeeze(H);                             % 실제 채널 생성 (NR x NT)

            % ---------- partial pilot observations (test-time) ----------
            H_pilot_lmmse_full = zeros(NR, NT);                             % partial->full LMMSE 결과 저장용
            for rcv = 1:NR
                noise = (randn(1, length(P_idx)) + 1i*randn(1, length(P_idx))) * sqrt(sigma2_ls/2); % AWGN for pilot obs
                % y_obs = H(rcv, P_idx) + noise;                              % 관찰된 파일럿 심볼들 (행벡터)
                % h_hat = lmmse_from_partial(y_obs.', P_idx, sigma2_ls);      % partial->full LMMSE (열벡터)

                % ====== Partial 관측도 동일 ======
                y_obs = zeros(1, length(P_idx));
                for l=1:N_L_test
                    noise = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(sigma2/2);
                    y_obs = y_obs + (H(rcv,P_idx) + noise);
                end
                y_obs = y_obs / N_L_test;
                h_hat = lmmse_from_partial(y_obs.', P_idx, sigma2/N_L_test);


                H_pilot_lmmse_full(rcv, :) = h_hat.';                       % 행벡터로 저장
            end

            % ---------- DNN 입력 구성 및 예측 ----------
            pilot_part_est = H_pilot_lmmse_full(:, P_idx);                  % estimate at pilot indices (NR x Npilot)
            x_test = [real(pilot_part_est(:)); imag(pilot_part_est(:))]';   % DNN 입력: row vector (1 x input_dim)
            x_test_norm = single((x_test - muX) ./ sX);                     % 정규화 및 single 변환
            y_pred = predict(net, x_test_norm);                             % DNN 예측 (1 x output_dim)

            % ---------- H_hat 재구성: pilot부는 LMMSE, null부는 DNN 예측 ----------
            H_hat = zeros(NR, NT);                                          % 복원된 채널 저장
            H_hat(:, P_idx) = H_pilot_lmmse_full(:, P_idx);                 % pilot 칼럼은 LMMSE로 채움

            half = output_dim/2;                                            % real/imag split index
            real_part = y_pred(1:half);                                     % 예측된 실수부
            imag_part = y_pred(half+1:end);                                 % 예측된 허수부
            complex_vec = real_part + 1i * imag_part;                       % complex vector (1 x output_dim/2)
            H_hat(:, N_idx) = reshape(complex_vec, NR, numel(N_idx));       % null 칼럼에 reshape하여 할당

            % ---------- 샘플별 NMSE 계산 및 누적 ----------
            e_dnn = norm(H - H_hat, 'fro')^2;                               % 복원 오차 제곱합
            pwr = norm(H, 'fro')^2;                                         % 원신호 전력
            tot_nmse_dnn = tot_nmse_dnn + e_dnn / (pwr + 1e-12);            % 정규화 NMSE 누적
        end
        NMSE_dnn(si) = tot_nmse_dnn / N_test;                               % 평균 NMSE 계산
        fprintf('r=%.3f EbN0=%2d dB  NMSE_DNN=%.4e  (done)\n', r, EbN0_db, NMSE_dnn(si)); % 진행 출력
    end

    results_dnn_map(r) = NMSE_dnn;                                          % 현재 r에 대한 NMSE 벡터 저장
    % net, muX, sX 등 저장 옵션 (필요 시)
    save(sprintf('dnn_r_%03d_net.mat', round(1000*r)), 'net', 'muX', 'sX'); % 학습 결과 저장 (파일명: dnn_r_875_net.mat 등)
end

%% ------------------ Plot results (NMSE vs Eb/N0) ------------------
figure('Color',[1 1 1],'Position',[100 100 900 600]);                       % figure 생성 및 크기 설정
semilogy(EbN0_dBs, results_baseline, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on; % baseline (full LMMSE) 플롯
markers = {'^-','>-','v-'};                                                  % proposed curve 마커 목록

ii = 1;
for r = r_list
    nmse_vec = results_dnn_map(r);                                          % DNN NMSE 벡터
    semilogy(EbN0_dBs, nmse_vec, markers{ii}, 'LineWidth', 1.2, 'MarkerSize', 6); % Proposed 플롯
    ii = ii + 1;
end

xlabel('E_b/N_0 [dB]','FontSize',15);                                        % x축 레이블 (영문)
ylabel('NMSE','FontSize',15);                                                 % y축 레이블 (영문)
title(sprintf('NMSE vs E_b/N_0 (NT=%d, NR=%d)', NT, NR),'FontSize',15);      % 타이틀 (영문)
grid on; box on;


% y 축 라벨 고정 코드
% ylim([1e-4 1e-1]);                                                          % y축 고정 (아래=1e-4, 위=1e-1)
% yticks([1e-1 1e-2 1e-3 1e-4]);                                              % (선택) 눈금 고정
% yticklabels({'10^{-1}','10^{-2}','10^{-3}','10^{-4}'});                   % (선택) 라벨 지정

legend_entries = [{'Baseline (Full LMMSE)'}];                                % 범례 초기화
for k = 1:numel(r_list)
    legend_entries{end+1} = sprintf('Proposed r=%.3f', r_list(k));           % Proposed 라인 이름 추가
end
legend(legend_entries, 'Location', 'southwest');                             % 범례 표시
set(gca, 'FontSize', 15);                                                    % 축 폰트 크기 설정



%% ------------------ Supporting functions (local functions define) ------------------

function H = gen_channel_peda(ns, NR, NT)
    % Generates ns samples of H under 3GPP Pedestrian A model
    % Input:
    %   ns  : number of channel samples
    %   NR  : number of receive antennas
    %   NT  : number of transmit antennas
    % Output:
    %   H   : (ns x NR x NT) MIMO channel tensor

    % --- 3GPP Pedestrian A (PedA) Power Delay Profile ---
    % delay = [0 0.11 0.19 0.41 0.61 1.73]*1e-6;                           % (초 단위) (참고)
    p_db  = [0 -1 -9 -10 -15 -20];                                         % (dB)
    p_lin = 10.^(p_db/10);                                                 % 선형 전력
    p_lin = p_lin / sum(p_lin);                                            % 정규화 (총합 = 1)

    % --- 출력 행렬 준비 ---
    H = zeros(ns, NR, NT);

    % --- 샘플 생성 ---
    for k = 1:ns
        Hk = zeros(NR, NT); % 한 샘플의 채널 행렬
        for rx = 1:NR
            for tx = 1:NT
                % PedA PDP에 따른 Rayleigh 페이딩 tap 생성
                taps = (randn(1, length(p_lin)) + 1i*randn(1, length(p_lin))) ...
                       .* sqrt(p_lin/2);
                % 등가 flat channel = 모든 tap의 합
                h_equiv = sum(taps);
                Hk(rx, tx) = h_equiv;
            end
        end
        H(k,:,:) = Hk;
    end
end


function h_hat = lmmse_from_partial_fn(y_obs, P_idx, sigma2, R_t)
    % 부분 관찰로부터 full 채널 LMMSE 추정 (partial->full LMMSE)
    % 입력:
    %   y_obs: M x 1 관찰 벡터 (pilot indices에서의 noisy observations)
    %   P_idx: 관찰된 칼럼 인덱스 (1 x M)
    %   sigma2: 노이즈 분산 (스칼라)
    %   R_t: transmit correlation matrix (NT x NT)
    % 출력:
    %   h_hat: NT x 1 complex vector (LMMSE로 복원된 full channel vector)
    P = P_idx(:).';                                                            % P를 행벡터로 정렬
    Rpp = R_t(P, P);                                                           % R_{pp} (M x M)
    Rtp = R_t(:, P);                                                           % R_{tp} (NT x M)
    A = Rpp + sigma2 * eye(length(P));                                         % A = Rpp + sigma2 I
    w = A \ y_obs;                                                             % w = A^{-1} y_obs
    h_hat = Rtp * w;                                                           % h_hat = Rtp * w (NT x 1)
end
