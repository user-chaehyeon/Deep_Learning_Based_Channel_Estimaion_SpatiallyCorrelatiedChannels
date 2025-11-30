%% BER / NMSE / Throughput simulation
clear; close all; clc; rng(0,'twister');
parallel.gpu.enableCUDAForwardCompatibility(true);

% % delete(findall(0)) ;  % plot 포함 학습진행 window 전체 닫기

%% # 안테나 변수 정의
NT   = 32;    % Tx antennas
NR   = 1;     % Rx antennas
NT_H = 8;
NT_V = 4;

assert( ...
    NT_H * NT_V == NT,'UPA size mismatch');

rho  = 0.9;   % nearest-neighbor TX corr.

r_list      = [0.5 0.625 0.75];                 % r = Npilot/NT
Np_list     = floor(NT * r_list);               % 버림 계산
Npilot_dict = containers.Map(r_list, Np_list);  % 두 값 매핑


%% # DNN 변수 정의
hidden         = [1024 1024 1024];
learning_rate  = 1e-5;
maxEpochs      = 30;
miniBatchSize  = 32;

DNN_train      = 2000;
valFraction    = 0.15;
N_test         = 200;           % Monte Carlo 채널 샘플 수(BER용)
Ns_data        = 500;           % 각 채널 샘플당 전송 심볼 수

mod_ord        = 4;             % QPSK(M = 4)
bits_p_sym     = log2(mod_ord); % log_2(4) = 2 bits 표현


%% # SNR : EbN0 & EsN0
EbN0_dB        = 0:5:20;   % 비트 단위 SNR(Eb/N0) 그리드 : 시뮬레이션 용
EbN0_range     = 0:5:20;   % 학습 중 선택할 SNR 범위
EbN0_train_dB  = EbN0_range(randi(length(EbN0_range)));

EsN0_dB         = EbN0_dB + 10*log10(bits_p_sym);   % convert Eb/N0 → Es/N0
Es_train = 10^((EbN0_train_dB + 10*log10(bits_p_sym))/10);
s2_tr    = 1/Es_train; %  noise variance 계산

NL_train = 8;          % DNN 학습에 사용될 파일럿 SNR 조정
NL_test  = 8;          % 실제 테스트 성능을 확인할 조건

log_every = max(1, floor(DNN_train/10)); % 진행 상황 출력


%% # Throughput 계산용 파라미터
N_sc         = 1024;   % 서브캐리어 개수
Nsym_per_sec = 15e3;   % 초당 OFDM 심볼 수
M_bits       = 2;      % 변조 비트수 (QPSK=2)
Denom        = 200;    % (1 - NP/Denom)

%% # UPA 공간 상관 행렬 (Kronecker)
Rh = toeplitz(rho.^(0:NT_H-1));
Rv = toeplitz(rho.^(0:NT_V-1));
Rt = kron(Rh,Rv); Rt = (Rt+Rt')/2;

% 고윳값 분해(Eigendecomposition) - R_t​=UDU^H
[U,D] = eig(Rt); lam = max(real(diag(D)),0);
Rt_sqrt = U*diag(sqrt(lam))*U';


%% # 채널 생성 핸들: PedA → TX correlation

function H = gen_PEDA(ns, NR, NT)
% Pedestrian-A (PedA) PDP 이용 - flat-fading MIMO 채널 생성
%   H  : 크기가 (ns x NR x NT) 인 복소 채널 텐서
%        H(k, rx, tx)는 k번째 스냅샷에서 rx번째 수신 안테나와
%        tx번째 송신 안테나 사이의 채널 계수 h_{rx,tx}

    p_db  = [0 -1 -9 -10 -15 -20];      % PedA PDP의 dB 값 (상대 전력)
    p_lin = 10.^(p_db/10); p_lin = p_lin/sum(p_lin);       % dB -> linear 변환 후, 전체 전력이 1이 되도록 정규화
    H = zeros(ns,NR,NT);        % 출력 채널 행렬 초기화: (ns x NR x NT)

    for k=1:ns      % k번째 스냅의 채널 (NR x NT)
        Hk = zeros(NR,NT);
        for rx=1:NR
            for tx=1:NT
                % 각 tap은 CN(0, p_lin(l)) 을 따르도록 생성
                % 실수/허수 각각 분산 p_lin/2 → 전체 분산 p_lin
                taps = (randn(1,length(p_lin))+1i*randn(1,length(p_lin))).*sqrt(p_lin/2);
                Hk(rx,tx) = sum(taps);
            end
        end
        H(k,:,:) = Hk;        % k번째 스냅샷 채널을 3차원 배열 H에 저장
    end
end

function Hc = apply_corr(H, Rt_sqrt)
% H : [ns x ? x ?] 채널 (PedA + i.i.d.)
% Rt_sqrt : [NT x NT] 송신 상관 행렬의 제곱근

    [ns, dim2, dim3] = size(H);       % 원래 H 의 크기 기준
    NT = size(Rt_sqrt,1);             % 송신 안테나 개수
    
    Hc = zeros(ns, dim2, dim3);       % 최종적으로 Hc 는 H 와 같은 크기를 유지
    
    for k = 1:ns
        Hw = squeeze(H(k,:,:));       % 크기: dim2 x dim3 또는 dim3 x dim2 중 하나일 수 있음
        
        if size(Hw,2) == NT           % Case 1: Hw 의 열 개수가 NT 이면 (NR x NT)
            Hcorr = Hw * Rt_sqrt;     % (NR x NT) * (NT x NT) = (NR x NT)

        elseif size(Hw,1) == NT       % Case 2: Hw 의 행 개수가 NT 이면 (NT x NR)
            Hcorr = Hw.' * Rt_sqrt;   % (NR x NT) * (NT x NT) = (NR x NT)
        
        else
            error('apply_corr: 예상치 못한 차원입니다. Hw=%s, Rt_sqrt=%s', ...
                mat2str(size(Hw)), mat2str(size(Rt_sqrt)));
        end
       
        % 여기서 Hcorr는 (NR x NT)
        % dim2 x dim3 = NR x NT 가 되도록 H 의 정의와 맞춰져 있어야 함
        Hc(k,:,:) = Hcorr;
    end
end


gen_channel = @(ns) apply_corr(gen_PEDA(ns,NR,NT), Rt_sqrt);


%% # LMMSE full & partial

% LMMSE(Low-complexity Linear Minimum Mean Square Error)  - 최적의 선형 채널 추정
% Perfect CSIR(Perfect Channel State Information at the Receiver)  - 수신기가 실제 채널 H 를 완벽하게 알고 있다는 가정
baseline_list = {'lmmse','perfect'}; 
N_nmse = 200;   % NMSE 계산용 채널 샘플 수

lmmse_full  = @(y, s2)  Rt*((Rt + s2*eye(NT))\y); 
%   y   : 수신된 파일럿 기반 관측 벡터 (NT x 1)
%   s2  : noise variance (sigma^2)
%   Rt  : Tx correlation matrix (NT x NT)
%   NT  : 송신 안테나 개수



lmmse_from_partial = @(yP, P, s2) lmmse_partial(yP,P,s2,Rt);
%   yP  : 파일럿을 보낸 안테나 subset P에서 얻어진 관측 벡터
%   P   : 파일럿을 보낸 송신 안테나 index 집합 (1 x Np)
%   s2  : noise variance (sigma^2)
%   Rt  : Tx correlation matrix (NT x NT)


%% # QPSK mod/demod (Gray
modQPSK   = @(bits) ( (1-2*bits(1:2:end)) + 1i*(1-2*bits(2:2:end)) )/sqrt(2); % QPSK 변조 (bits → symbols)
%   bits : 길이가 2의 배수인 비트 벡터 (0 또는 1),   bit 0 → +1 ,  bit 1 → -1
%   symbols : QPSK 심볼 벡터 (complex),    QPSK = (I + jQ)/sqrt(2)


demodQPSK = @(z) [real(z)<0; imag(z)<0];  % QPSK 복조 (symbols → bits)
%   z : 수신된 QPSK 심볼 (complex)
%   bits : 복원된 비트 벡터 (2 x length(z))
%   real(z) < 0 → 1,  real(z) >= 0 → 0
%   imag(z) < 0 → 1,  imag(z) >= 0 → 0


%% # 파일럿 인덱스 생성 함수 핸들
make_pilots = @(NT_H,NT_V,Np) uniform(NT_H,NT_V,Np);
%   NT_H : 수평(horizontal) 방향 안테나 개수
%   NT_V : 수직(vertical) 방향 안테나 개수
%   Np   : 파일럿을 보낼 송신 안테나 개수





%% === Train one DNN per r
% net   : r값 별로 학습된 DNN 네트워크 저장
% stats : r값 별 성능 통계(NMSE, BER, Throughput) 저장
nets  = containers.Map('KeyType','double','ValueType','any');
stats = containers.Map('KeyType','double','ValueType','any');

%% 1. DNN 네트워크들을 r 값별로 저장
for r = r_list
    Np = Npilot_dict(r);                    % 파일럿 안테나 수
    [P_idx, N_idx] = uniform(NT_H,NT_V,Np);

    %% a. 입력/출력 차원 계산 ----
    % 채널 행렬 H는 복소수 이고, 딥러닝 네트워는 실수 기반이므로 Re, Im을 분해
    % feature dimension이 2배가 되어 2를 곱함
    inDimension  = 2*NR*Np;                   % DNN의 입력차원 : (Re/Im) × NR × Np -> 파일럿 안테나에 대한 복원 채널
    outDimension = 2*NR*(NT-Np);              % DNN이 예측해야하는 출력차원 : (Re/Im) × NR × (NT-Np) -> 파일럿 없는 안테나 채널

    %% b. 학습 데이터 메모리 할당
    % DNN_train : DNN 학습에 사용할 샘플 개수
    nSamples = round(DNN_train);                      % 실수일 때 대비, 학습 데이터 개수를 정수로 반올림해서 확정
    Xtr = zeros(nSamples, inDimension,  'single');    % 학습 입력 데이터를 저장할 nSamples × inDim 행렬
    Ytr = zeros(nSamples, outDimension, 'single');    % 학습 정답(레이블) 데이터를 저장할 nSamples × outDim 행렬

    %% c. 한 샘플 n에 대해 - 학습 SNR 생성

    for n = 1:nSamples
        % 실제 채널 생성
        H = squeeze(gen_channel(1));           % 크기: (NR x NT) 또는 (NT x NR)
        H = reshape(H, NR, NT);                % 강제로 (NR x NT)로 맞춤
        h_vec = H(:);                          % 길이: NR*NT - 전체 채널 벡터

        % partial pilots → LMMSE from partial
        H_pilot = H;                           % 크기 맞춰진 실제 채널
        yP = zeros(length(P_idx),1);           % 

        % Partial pilot 관측 생성
        for l = 1:NL_train
            nNoise = (randn(1,length(P_idx)) + 1i*randn(1,length(P_idx))) * sqrt(s2_tr/2);
            yP = yP + (H_pilot(1,P_idx).' + nNoise.');
        end

        yP = yP / NL_train;  % yP는 hP+n에서 SNR을 고려한 평균화 값

        % 부분 파일럿 partial LMMSE로부터 full-length LMMSE 추정
        % HfullP는 "부분 파일럿 + LMMSE"로 복원한 full 채널 (노이즈 포함 추정값)
        hhat = lmmse_partial(yP.', P_idx, s2_tr/NL_train, Rt);          % (NT x 1)
        HfullP = reshape(hhat, NR, NT);                                 % (NR x NT)

        % DNN 입력/타겟 구성
        Xin    = HfullP(:, P_idx);          % (NR x Np)
        target = h_vec(N_idx);              % ((NT-Np)*NR x 1)  (NR=1이면 (NT-Np) x 1)

        % 실수/허수 분해 후 행벡터로
        xvec = [real(Xin(:));    imag(Xin(:))   ].';  % 1 x inDimension : 파일럿 안테나에 대한 채널 추정값
        yvec = [real(target(:)); imag(target(:))].';  % 1 x outDimension : 실제 채널에서 null 안테나 인덱스에 해당하는 성분만 모아둔 것

        Xtr(n,:) = single(xvec);
        Ytr(n,:) = single(yvec);

        if mod(n, log_every) == 0
            fprintf('[TrainSet r=%.3f] %5d / %5d (%.1f%%)\n', ...
                    r, n, nSamples, 100*n/nSamples);
        end

    end

    % 각 feature마다 평균/분산으로 정규화
    muX = mean(Xtr,1); sX = std(Xtr,0,1) + 1e-12;
    muY = mean(Ytr,1); sY = std(Ytr,0,1) + 1e-12;
    XtrN = single((Xtr - muX)./sX);
    YtrN = single((Ytr - muY)./sY);

    % ---- Train / Val split ----
    ntr    = floor((1 - valFraction)*nSamples);
    XTrain = XtrN(1:ntr,:);      YTrain = YtrN(1:ntr,:);
    XVal   = XtrN(ntr+1:end,:);  YVal   = YtrN(ntr+1:end,:);

    % ---- DNN 구조 (hidden) ----
    layers = [
        featureInputLayer(inDimension,'Normalization','none')
    ];

    for i = 1:numel(hidden)
        layers = [
            layers
            fullyConnectedLayer(hidden(i))
            reluLayer
        ];
    end

    layers = [
        layers
        fullyConnectedLayer(outDimension)
        regressionLayer
    ];

    opts = trainingOptions('adam', ...
        'InitialLearnRate',   learning_rate, ...
        'MaxEpochs',          maxEpochs, ...
        'MiniBatchSize',      miniBatchSize, ...
        'Shuffle',            'every-epoch', ...
        'ValidationData',     {XTrain, YTrain}, ...  %  XVal,YVal
        'Verbose',            true, ...
        'Plots',              'training-progress', ...
        'ExecutionEnvironment','auto');

    % net = trainNetwork(XTrain, YTrain, layers, opts); % 검증 세트를 안 쓰는 상태
    net = trainNetwork(XVal, YVal, layers, opts);

    % 각 r마다 학습된 네트워크와 정규화 통계를 map에 저장
    nets(r)  = net;
    stats(r) = struct('P_idx',P_idx,'N_idx',N_idx, ...
                      'muX',muX,'sX',sX,'muY',muY,'sY',sY);

    fprintf('DNN 학습 완료 (r = %.3f)\n', r);
end


%% # #  BER simulation

% % 매 SNR 포인트에 대해:
% % Baseline: 모든 안테나가 파일럿 (full-pilot LMMSE / 또는 실제채널 사용)
% % Proposed: 각 r 에 대해 :
% % 부분 파일럿 + LMMSE → HfullP
% % DNN으로 N_idx 채널 복원 → Hhat
% % MRT precoder w로 단일 스트림 전송, QPSK 변조, qammod/demod 사용
% % 심볼 수 Ns_data 만큼 Monte Carlo, BER 집계


BER_base = containers.Map();
for bb = 1:length(baseline_list)
    BER_base(baseline_list{bb}) = zeros(numel(EbN0_dB),1);
end

BER_map  = containers.Map('KeyType','double','ValueType','any');
for r = r_list
    BER_map(r) = zeros(numel(EbN0_dB),1);
end

for si = 1:numel(EbN0_dB)
    Es_lin = 10^(EsN0_dB(si)/10);
    s2     = 1/Es_lin;        % 잡음 분산

    %% ---- Baseline: full-pilot ----
    for bb = 1:length(baseline_list)
        base_type = baseline_list{bb};
        bitErr_C = 0; bitTot_C = 0;

        for tcase = 1:N_test
            % ---- 채널 생성 및 정렬 ----
            H = gen_channel(1);
            H = ensure(H, NR, NT);   % (1 x NT)

            % ---- Baseline 채널 추정 Hc ----
            Hc = zeros(NR, NT);

            switch base_type
                case 'lmmse'
                    for rx = 1:NR
                        y_p = zeros(1, NT);
                        for l = 1:NL_test
                            n = (randn(1,NT) + 1i*randn(1,NT))*sqrt(s2/2);
                            y_p = y_p + (H(rx,:) + n);
                        end
                        y_p  = y_p / NL_test;
                        hhat = lmmse_full(y_p.', s2/NL_test);   % (NT x 1)
                        Hc(rx,:) = hhat.';                          % (1 x NT)
                    end
                case 'perfect'
                    Hc = H;    % 이상적 CSIR
                otherwise
                    error('Unknown baseline_type: %s', base_type);
            end

            H  = ensure(H,  NR, NT);
            Hc = ensure(Hc, NR, NT);

            % ---- 데이터 비트 및 심볼 생성 ----
            M     = mod_ord;
            kbits = bits_p_sym;
            bits  = randi([0 1], kbits*Ns_data, 1);

            X      = reshape(bits, kbits, []).';
            symInt = bi2de(X, 'left-msb');
            s      = qammod(symInt, M, 'gray').';
            s      = s ./ sqrt(mean(abs(s).^2));

            % ---- MRT beamforming ----
            H_row  = reshape(H(1,:),  1, []);
            Hc_row = reshape(Hc(1,:), 1, []);

            [w, g_hat] = precoder_MRT(Hc_row);

            for k = 1:Ns_data
                n = (randn(NR,1) + 1i*randn(NR,1))*sqrt(s2/2);
                y = H_row * w * s(k) + n;

                rhat = y * conj(g_hat) / (abs(g_hat)^2 + 1e-12);

                bits_hat = qamdemod(rhat, M, 'gray', 'OutputType','bit');

                idx_start = (k-1)*kbits + 1;
                idx_end   = k*kbits;
                bitErr_C  = bitErr_C + sum(bits_hat(:) ~= bits(idx_start:idx_end));
                bitTot_C  = bitTot_C + kbits;
            end
        end

        tmp = BER_base(base_type);
        tmp(si) = bitErr_C / bitTot_C;
        BER_base(base_type) = tmp;
    end




    %% ---- Proposed (DNN 기반 채널 추정, 각 r에 대해) ----
    for r = r_list
        cfg_r = stats(r);
        net   = nets(r);

        P_idx = cfg_r.P_idx;
        N_idx = cfg_r.N_idx;
        muX   = cfg_r.muX;
        sX    = cfg_r.sX;
        muY   = cfg_r.muY;
        sY    = cfg_r.sY;

        bitErr = 0; bitTot = 0;

        for tcase = 1:N_test
            % ---- 실제 채널 ----
            H2 = gen_channel(1);
            H2 = ensure(H2, NR, NT);   % (1 x NT)

            % ---- partial pilot -> LMMSE from partial (full-length) ----
            HfullP = zeros(NR, NT);
            for rx = 1:NR
                yP = zeros(length(P_idx),1);
                for l = 1:NL_test
                    n = (randn(1,length(P_idx)) + 1i*randn(1,length(P_idx)))*sqrt(s2/2);
                    yP = yP + (H2(rx,P_idx).' + n.');
                end
                yP = yP/NL_test;
                hhat = lmmse_partial(yP.', P_idx, s2/NL_test, Rt);
                HfullP(rx,:) = hhat.';    % (1 x NT)
            end

            % ---- DNN을 이용해 null 안테나 채널 복원 ----
            Xin = HfullP(:,P_idx);
            xin = [real(Xin(:)); imag(Xin(:))]';
            xinN = single((xin - muX)./sX);

            ypredN = predict(net, xinN);
            half   = numel(ypredN)/2;
            chanN  = (ypredN(1:half).*sY(1:half) + muY(1:half)) + ...
                1i*(ypredN(half+1:end).*sY(half+1:end) + muY(half+1:end));
            Chan = reshape(chanN, NR, numel(N_idx));

            % 최종 DNN 기반 채널 추정 Hhat
            Hhat = HfullP;
            Hhat(:,N_idx) = Chan;           % (1 x NT)
            Hhat = ensure(Hhat, NR, NT);

            % ---- 데이터 비트 및 심볼 생성 ----
            M     = mod_ord;
            kbits = bits_p_sym;
            bits  = randi([0 1], kbits*Ns_data, 1);

            X      = reshape(bits, kbits, []).';
            symInt = bi2de(X, 'left-msb');
            s      = qammod(symInt, M, 'gray').';
            s      = s ./ sqrt(mean(abs(s).^2));

            % ---- MRT precoder (Hhat 기반) ----
            H2_row   = reshape(H2(1,:),   1, []);
            Hhat_row = reshape(Hhat(1,:), 1, []);

            [w_hat, g_hat] = precoder_MRT(Hhat_row);
          
            % ---- MRT 기반 데이터 전송/복조 ----
            for k = 1:Ns_data
                % 수신: y = H2_row * w_hat * s(k) + n
                n = (randn(NR,1) + 1i*randn(NR,1))*sqrt(s2/2);
                y = H2_row * w_hat * s(k) + n;

                % 등화: rhat = y * conj(g_hat) / |g_hat|^2
                rhat = y * conj(g_hat) / (abs(g_hat)^2 + 1e-12);

                bits_hat = qamdemod(rhat, M, 'gray', 'OutputType','bit');

                idx_start = (k-1)*kbits + 1;
                idx_end   = k*kbits;
                bitErr    = bitErr + sum(bits_hat(:) ~= bits(idx_start:idx_end));
                bitTot    = bitTot + kbits;
            end
        end

        v = BER_map(r);
        v(si) = bitErr / bitTot;
        BER_map(r) = v;
    end


    % 진행 상황 출력
    v1 = BER_map(r_list(1));
    v2 = BER_map(r_list(2));
    v3 = BER_map(r_list(3));

    BER_L = BER_base('lmmse');
    BER_P = BER_base('perfect');

    fprintf(['SNR=%2d dB | BER(LMMSE)=%.3e, BER(Perfect)=%.3e ', ...
        '| proposed r=%.3f/%.3f/%.3f -> %.3e / %.3e / %.3e\n'], ...
        EbN0_dB(si), ...
        BER_L(si), BER_P(si), ...
        r_list(1), r_list(2), r_list(3), ...
        v1(si), v2(si), v3(si));
end



%% # #  Throughput in bps

% % N_P : 파일럿 안테나 개수
% % M_bits : 변조 비트 수
% % N_sc, Nsym_per_sec : 서브캐리어 수, 초당 OFDM 심볼 수
% % Conventional: NP_conv = NT
% % Proposed: NP_r = Npilot_dict(r)

% Conventional (baseline): NP = NT
NP_conv        = NT;
data_frac_conv = max(0, 1 - NP_conv/Denom);

TH_base = containers.Map();
for bb = 1:length(baseline_list)
    bname = baseline_list{bb};
    BERv  = BER_base(bname);
    TH_base(bname) = data_frac_conv .* (1 - BERv) .* M_bits .* N_sc .* Nsym_per_sec;
end

TH_conv_bps = TH_base('lmmse');

% Proposed (각 r)
TH_map_bps = containers.Map('KeyType','double','ValueType','any');
for r = r_list
    NP_r       = Npilot_dict(r);
    data_frac_r= max(0, 1 - NP_r/Denom);
    BERv       = BER_map(r);
    TH_map_bps(r) = data_frac_r .* (1 - BERv) .* M_bits .* N_sc .* Nsym_per_sec;
end

%% [ Plot ] Throughput vs Eb/N0
figure('Color','w','Position',[100 100 900 600]); 
hold on; grid on; box on;

mk = {'-o','-^','-s','-d','->','-v','-<','-p','-h'};
colors = lines(2);   % baseline 2개용 색상
leg = {};

% --- Baseline: LMMSE ---
plot(EbN0_dB, TH_base('lmmse')/1e7, mk{1}, ...
     'LineWidth',1.5,'MarkerSize',6,'Color',colors(1,:));
leg{end+1} = 'Baseline (LMMSE)';

% --- Baseline: Perfect CSIR ---
plot(EbN0_dB, TH_base('perfect')/1e7, mk{2}, ...
     'LineWidth',1.5,'MarkerSize',6,'Color',colors(2,:), ...
     'LineStyle','--');
leg{end+1} = 'Baseline (Perfect CSIR)';

% --- Proposed (각 r) ---
for i = 1:numel(r_list)
    r = r_list(i);
    plot(EbN0_dB, TH_map_bps(r)/1e7, mk{2+i}, ...
        'LineWidth',1.2,'MarkerSize',6);
    leg{end+1} = sprintf('Proposed (r = %.3f)', r);
end

xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('Throughput [\times10^7 bps]','FontSize',16);
title(sprintf('Throughput vs E_b/N_0 (N_T=%d, N_R=%d, M=%d)', ...
      NT, NR, M_bits),'FontSize',16);
legend(leg, 'Location','southeast','FontSize',12);

hold off;

%% [ Plot ] BER
figure('Color','w','Position',[100 100 900 600]);
hold on; grid on; box on;

mk = {'-o','-^','-s','-d','->','-v','-<','-p','-h'};
colors = lines(2);
leg = {};

% --- Baseline: LMMSE ---
semilogy(EbN0_dB, BER_base('lmmse'), mk{1}, ...
    'LineWidth',1.5,'MarkerSize',6,'Color',colors(1,:));
leg{end+1} = 'Baseline (LMMSE)';

% --- Baseline: Perfect CSIR ---
semilogy(EbN0_dB, BER_base('perfect'), mk{2}, ...
    'LineWidth',1.5,'MarkerSize',6,'Color',colors(2,:), ...
    'LineStyle','--');
leg{end+1} = 'Baseline (Perfect CSIR)';

% --- Proposed (각 r) ---
for i = 1:numel(r_list)
    r = r_list(i);
    v = BER_map(r);
    semilogy(EbN0_dB, v, mk{2+i}, ...
        'LineWidth',1.2,'MarkerSize',6);
    leg{end+1} = sprintf('Proposed (r = %.3f)', r);
end

xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('BER','FontSize',16);
title(sprintf('BER vs E_b/N_0 (N_T=%d, N_R=%d)',NT,NR),'FontSize',16);
legend(leg, 'Location','southwest','FontSize',12);

hold off;


%% # #  NMSE simulation (LMMSE vs DNN)

% % 매 SNR 포인트마다 N_nmse 개의 채널에 대해
% % Baseline: full-pilot LMMSE 로 Hc 추정
% % Proposed: partial-pilot + DNN 으로 Hhat 생성

NMSE_base = containers.Map();
for bb = 1:length(baseline_list)
    NMSE_base(baseline_list{bb}) = zeros(numel(EbN0_dB),1);
end

results_dnn_map  = containers.Map('KeyType','double','ValueType','any');
for r = r_list
    results_dnn_map(r) = zeros(numel(EbN0_dB),1);
end

for si = 1:numel(EbN0_dB)
    Es_lin = 10^(EsN0_dB(si)/10);
    s2     = 1/Es_lin;

    %% --- Baseline NMSE (LMMSE, Perfect 각각) ---
    for bb = 1:length(baseline_list)
        base_type = baseline_list{bb};
        nmse_sum = 0;

        for tcase = 1:N_nmse
            % 실제 채널
            H_true = squeeze(gen_channel(1));
            H_true = reshape(H_true, NR, NT);

            % 추정 채널 Hc
            Hc = zeros(NR,NT);
            switch base_type
                case 'lmmse'
                    for rx=1:NR
                        y = zeros(1,NT);
                        for l=1:NL_test
                            n = (randn(1,NT)+1i*randn(1,NT))*sqrt(s2/2);
                            y = y + (H_true(rx,:) + n);
                        end
                        y = y/NL_test;
                        hhat = lmmse_full(y.', s2/NL_test);  % NT x 1
                        Hc(rx,:) = hhat.';
                    end
                case 'perfect'
                    Hc = H_true;
                otherwise
                    error('Unknown baseline_type: %s', base_type);
            end

            nmse_sum = nmse_sum + ...
                norm(Hc(:)-H_true(:))^2 / (norm(H_true(:))^2 + eps);
        end

        v = NMSE_base(base_type);
        v(si) = nmse_sum / N_nmse;
        NMSE_base(base_type) = v;
    end

    %% --- Proposed NMSE (partial-pilot + DNN) ---
    for r = r_list
        nmse_dnn = 0;

        for tcase = 1:N_nmse
            % 실제 채널
            H_true = squeeze(gen_channel(1));
            H_true = reshape(H_true, NR, NT);

            % partial pilot + LMMSE
            HfullP = zeros(NR,NT);
            for rx=1:NR
                yP = zeros(length(P_idx),1);
                for l=1:NL_test
                    n = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(s2/2);
                    yP = yP + (H_true(rx,P_idx).' + n.');
                end
                yP = yP / NL_test;

                hhatP = lmmse_partial(yP.', P_idx, s2/NL_test, Rt);
                HfullP(rx,:) = hhatP.';   % 1 x NT
            end

            % DNN 보정
            Xin = HfullP(:,P_idx);                   % NR x Np
            xin = [real(Xin(:)); imag(Xin(:))]';     % 1 x inDim
            xinN = single((xin - muX)./sX);

            ypredN = predict(net, xinN);
            half = numel(ypredN)/2;

            chanN = (ypredN(1:half).*sY(1:half) + muY(1:half)) + ...
                1i*(ypredN(half+1:end).*sY(half+1:end) + muY(half+1:end));
            Chan = reshape(chanN, NR, numel(N_idx)); % NR x (NT-Np)

            Hhat = HfullP;
            Hhat(:,N_idx) = Chan;

            nmse_tmp = norm(Hhat(:)-H_true(:))^2 / (norm(H_true(:))^2 + eps);
            nmse_dnn = nmse_dnn + nmse_tmp;
        end

        v = results_dnn_map(r);
        v(si) = nmse_dnn / N_nmse;
        results_dnn_map(r) = v;
    end

    fprintf('NMSE: SNR=%2d dB done.\n', EbN0_dB(si));
end

%% ------ Plot NMSE (dB) ----------------
figure('Color','w','Position',[100 100 900 600]);
hold on; grid on; box on;

mk = {'-o','-^','-s','-d','->','-v','-<','-p','-h'};
colors  = lines(2);
leg = {};

% Baseline (LMMSE)
plot(EbN0_dB, 10*log10(NMSE_base('lmmse')), mk{1}, ...
     'LineWidth',1.5,'MarkerSize',6,'Color',colors(1,:));
leg{end+1} = 'Baseline (LMMSE)';

% Baseline (Perfect CSIR)
plot(EbN0_dB, 10*log10(NMSE_base('perfect')), mk{2}, ...
     'LineWidth',1.5,'MarkerSize',6,'Color',colors(2,:), ...
     'LineStyle','--');
leg{end+1} = 'Baseline (Perfect CSIR)';

% Proposed (r별)
for i = 1:numel(r_list)
    r = r_list(i);
    v_lin = results_dnn_map(r);
    plot(EbN0_dB, 10*log10(v_lin), mk{2+i}, ...
        'LineWidth',1.2, 'MarkerSize',6);
    leg{end+1} = sprintf('Proposed (r = %.3f)', r);
end

xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('NMSE [dB]','FontSize',16);
title(sprintf('NMSE vs E_b/N_0 (N_T=%d, N_R=%d)',NT,NR),'FontSize',16);
legend(leg,'Location','southwest','FontSize',12);
hold off;

%% [ Plot ] NMSE (dB 단위)
figure('Color','w','Position',[100 100 900 600]);
hold on; grid on; box on;

mk = {'-o','-^','-s','-d','->','-v','-<','-p','-h'};
colors  = lines(2);
leg = {};

% --- Baseline (LMMSE) ---
plot(EbN0_dB, 10*log10(NMSE_base('lmmse')), mk{1}, ...
     'LineWidth',1.5,'MarkerSize',6,'Color',colors(1,:));
leg{end+1} = 'Baseline (LMMSE)';

% --- Baseline (Perfect CSIR) ---
plot(EbN0_dB, 10*log10(NMSE_base('perfect')), mk{2}, ...
     'LineWidth',1.5,'MarkerSize',6,'Color',colors(2,:), ...
     'LineStyle','--');
leg{end+1} = 'Baseline (Perfect CSIR)';

% --- Proposed (각 r) ---
for i = 1:numel(r_list)
    r = r_list(i);
    v_lin = results_dnn_map(r);   % 선형 스케일 NMSE
    plot(EbN0_dB, 10*log10(v_lin), mk{2+i}, ...
        'LineWidth',1.2, 'MarkerSize',6);
    leg{end+1} = sprintf('Proposed (r = %.3f)', r);
end

xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('NMSE [dB]','FontSize',16);
title(sprintf('NMSE vs E_b/N_0 (N_T=%d, N_R=%d)',NT,NR),'FontSize',16);
legend(leg,'Location','southwest','FontSize',12);

hold off;

%% [ Plot ] Throughput gain over Conventional (percentage)
eps0 = 1e-12;
base = max(TH_conv_bps, eps0);

GAIN_map_pct = containers.Map('KeyType','double','ValueType','any');
for i = 1:numel(r_list)
    r = r_list(i);
    TH_r = TH_map_bps(r);
    GAIN_map_pct(r) = 100 * (TH_r - TH_conv_bps) ./ base;
end

% ---- Plot: % improvement vs E_b/N_0 ----
figure('Color','w','Position',[120 120 900 600]);
hold on; grid on; box on;

mk = {'-o','-^','-s','-d','->','-v','-<','-p','-h'};

% 기준선 0%
plot(EbN0_dB, zeros(size(EbN0_dB)), '--', 'LineWidth', 1.0);
leg = {};
leg{end+1} = 'Baseline (0%)';

% Proposed (각 r)
for i = 1:numel(r_list)
    r = r_list(i);
    plot(EbN0_dB, GAIN_map_pct(r), mk{i}, ...
         'LineWidth',1.5, 'MarkerSize',6);
    leg{end+1} = sprintf('Proposed (r = %.3f)', r);
end

xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('Throughput improvement over Conventional [%]','FontSize',16);
title(sprintf('Throughput Gain vs E_b/N_0 (N_T=%d, N_R=%d, M=%d)', ...
      NT, NR, M_bits), 'FontSize',16);
legend(leg, 'Location','southeast','FontSize',12);

hold off;



%% ===== Local functions =====


function H2 = ensure(H, NR, NT)
    %   입력 H 를 항상 (NR x NT) 크기의 행렬로 맞추는 보조 함수

    H2 = squeeze(H);         % 차원 3→2로 축소 시도
    if ~ismatrix(H2)
        error('ensure: H must be 2D after squeeze. 현재 size(H) = [%s]', ...
              num2str(size(H2)));
    end

    [r,c] = size(H2);

    if r == NR && c == NT        % Case 1: 이미 (NR x NT)
        return;
   
    elseif r == NT && c == NR    % Case 2: (NT x NR)이면 transpose
        H2 = H2.';
        return;
   
    elseif numel(H2) == NR*NT    % Case 3: 원소 개수가 맞으면 reshape
        H2 = reshape(H2, NR, NT);
        return;
    else
        error('ensure: 예상치 못한 크기입니다. size(H) = [%d %d], NR=%d, NT=%d', ...
              r, c, NR, NT);
    end
end




function h_hat = lmmse_partial(y_obs, P_idx, sigma2, Rt)
    % lmmse_partial
    %  - 부분 파일럿 관측 y_obs 와 송신 측 상관 행렬 Rt 이용
    %    전체 송신 안테나 채널 벡터 h 를 LMMSE 방식으로 추정
    %
    % 가정:
    %   h ~ CN(0, Rt) 인 송신 채널 벡터 (NT x 1)
    %   y_P = h_P + n,  n ~ CN(0, sigma2 * I),  P는 파일럿을 보낸 안테나 집합
    %
    % 입력:
    %   y_obs  : 파일럿을 보낸 안테나 위치 P_idx에서 관측한 수신 신호 벡터
    %            크기는 (1 x |P|) 또는 (|P| x 1) 로 간주
    %            요소는 y_obs(p) = h(p) + noise(p) 에 해당
    %   P_idx  : 파일럿을 보낸 송신 안테나 인덱스 집합 (indices of pilot antennas)
    %            예) P_idx = [1 5 9 13 ...]
    %   sigma2 : 노이즈 분산 σ^2 (noise variance)
    %   Rt     : 송신 안테나 간 상관 행렬 (Transmit correlation matrix, NT x NT)
    %
    % 출력:
    %   h_hat  : 전체 송신 안테나에 대한 LMMSE 추정 채널 벡터 (NT x 1)
    %
    % LMMSE 수식:
    %   관측: y_P = h_P + n,  h ~ CN(0, Rt),  n ~ CN(0, sigma2 I)
    %   R_pp = Rt(P,P),  R_tp = Rt(:,P)
    %
    %   h_hat = R_tp * (R_pp + sigma2 I)^(-1) * y_P
    %         = Rtp * A^{-1} * y_obs
    %
    % 여기서:
    %   - R_pp (Rpp) : 파일럿 안테나들끼리의 상관 행렬 (|P| x |P|)
    %   - R_tp (Rtp) : 전체 안테나 ↔ 파일럿 안테나 간 상관 행렬 (NT x |P|)
    %   - y_obs      : 길이 |P|의 관측 벡터

    P   = P_idx(:).';        % 파일럿 인덱스를 행 벡터 형태로 정리
    Rpp = Rt(P,P);          % 파일럿 안테나들 사이의 상관 행렬
    Rtp = Rt(:,P);          % 전체 안테나 ↔ 파일럿 안테나 간 상관
    A   = Rpp + sigma2*eye(length(P));      % (관측 공분산 + 노이즈 공분산)
    w   = A \ y_obs.';       % (|P| x 1)
    h_hat = Rtp * w;         % (NT x 1)
end

function [P_idx, N_idx] = uniform(NH,NV,Np)
    % uniform
    %  - 2D UPA(Uniform Planar Array) 구조에서 파일럿을 보내는 안테나 인덱스를
    %    "가능한 한 균일하게" 선택
    %
    % 입력:
    %   NH : 수평(horizontal) 방향 안테나 개수
    %   NV : 수직(vertical) 방향 안테나 개수
    %   Np : 선택하고자 하는 파일럿 안테나 개수
    %
    % 출력:
    %   P_idx : 파일럿 안테나 인덱스 집합 (1차원 인덱스, 길이 Np)
    %           2D 인덱스 (h, v)를 1D 인덱스로 매핑 시 index = (v-1)*NH + h 사용
    %   N_idx : 파일럿이 아닌 나머지 안테나 인덱스 집합
    %
    % 동작 :
    %   1) NH x NV 격자(UPA)에서 수평/수직 방향으로 linspace를 이용해 간격을 두고
    %      후보 위치 생성
    %   2) 만들어진 격자에서 Np개 이상이면, 균일하게 Np개를 subsampling
    %   3) Np개보다 적으면, 나머지를 다른 위치에서 골라와서 총 Np개가 되도록 채움
    %   4) P_idx 는 정렬된(pilot) 인덱스, N_idx 는 그 외 인덱스

    % 수평/수직 방향으로 각각 몇 개의 그리드를 사용할지 결정
    NHg = min(NH,Np);
    NVg = ceil(Np/NHg);
    if NVg>NV
        NVg = NV; 
        NHg = min(NH,ceil(Np/NVg));
    end

    % 수평/수직 방향 인덱스를 균일 선택
    hs = unique(round(linspace(1,NH,NHg)),'stable');
    vs = unique(round(linspace(1,NV,NVg)),'stable');

    % 2D 그리드 (HH, VV) 생성
    [HH,VV] = meshgrid(hs,vs);
    % 2D 인덱스를 1D 인덱스로 변환: (v-1)*NH + h
    P = reshape((VV-1)*NH + HH,1,[]);
    if numel(P)>=Np
        % 후보가 Np개 이상이면, 전체 후보 P에서 균일하게 Np개만 선택
        pick  = round(linspace(1,numel(P),Np));
        P_idx = sort(P(pick));
    else
        % 후보가 Np개보다 적으면, 나머지는 다른 위치에서 추가로 선택
        need  = Np-numel(P);
        pool  = setdiff(1:NH*NV,P);     % 아직 안 쓴 안테나 인덱스 집합
        P_idx = sort([P, pool(round(linspace(1,numel(pool),need)))]);
    end

    % N_idx : 파일럿이 아닌 나머지 안테나들
    N_idx = setdiff(1:NH*NV, P_idx);
end

function [w, g_hat] = precoder_MRT(h_est)
% MRT precoder 생성 함수
% 입력:
%   h_est : 1 x NT 추정 채널 (row vector) 라고 "기대"하지만
%           실제로는 (NT x 1) 이거나 다른 모양일 수도 있음
%
% 출력:
%   w     : NT x 1 MRT precoder
%   g_hat : 등가 채널 스칼라 = h_est * w

    % 1) 우선 vec으로 펼친 뒤, 항상 "행 벡터(1 x NT)"로 만든다
    h_row = reshape(h_est, 1, []);     % 1 x NT

    % 2) MRT precoder: w = h_row^H / ||h_row||
    w = (h_row.');                     % NT x 1
    w = w ./ (norm(h_row) + 1e-12);    % NT x 1, 정규화

    % 3) 등가 채널: g_hat = h_row * w     (1xNT * NT x1 = 1x1)
    g_hat = h_row * w;
end
