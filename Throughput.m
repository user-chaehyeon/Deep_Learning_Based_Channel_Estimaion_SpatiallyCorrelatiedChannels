%% ===== BER simulation (Fig.7-style) =====
clear; close all; clc; rng(0,'twister');

%% ---------------- System / Paper params ----------------
NT   = 64;          % Tx antennas  (논문 Fig.7은 128; 원하면 128로 바꾸고 Npilot_dict도 맞춰 주세요)
NR   = 2;           % Rx antennas
NT_H = 8; NT_V = 8; assert(NT_H*NT_V==NT,'UPA size mismatch');

rho  = 0.9;         % nearest-neighbor TX corr.
r_list = [0.875 0.938 0.969];
Npilot_dict = containers.Map([0.875 0.938 0.969],[56 60 62]);  % |P|


%% ---------------- DNN / data settings ----------------
hidden_units  = [1024 1024 1024];
learning_rate = 1e-5;
maxEpochs     = 30;
miniBatchSize = 32;

N_train     = 2000;    % DNN train samples
valFraction = 0.15;
N_test      = 200;     % Monte Carlo 채널 샘플 수 (BER에서는 심볼 수가 커서 200 정도면 충분)
Ns_data     = 800;     % 각 채널 샘플당 전송 심볼 수(안테나 하나씩 순번 전송)

%% ---------------- SNR / repetition ----------------
EbN0_train_dB = 40;
mod_order       = 4;     % QPSK
bits_per_symbol = log2(mod_order);  % =2
EbN0_dB   = [0 5 10 15 20];
EsN0_dBs  = EbN0_dB + 10*log10(bits_per_symbol);

NL_train = 8;            % repetition for pilots (avg)
NL_test  = 8;


% [맨 위] 로그 주기
log_every = max(1, floor(N_train/10));

%% ---------------- TX corr Rt (UPA Kronecker) ----------------
Rh = toeplitz(rho.^(0:NT_H-1));
Rv = toeplitz(rho.^(0:NT_V-1));
Rt = kron(Rh,Rv); Rt = (Rt+Rt')/2;
[U,D] = eig(Rt); lam = max(real(diag(D)),0);
Rt_sqrt = U*diag(sqrt(lam))*U';

%% ---------------- Helpers ----------------
gen_channel = @(ns) apply_corr(gen_channel_peda(ns,NR,NT), Rt_sqrt); % (ns x NR x NT)
lmmse_full  = @(y,s2)  Rt*((Rt + s2*eye(NT))\y);
lmmse_from_partial = @(yP,P,s2) lmmse_from_partial_fn(yP,P,s2,Rt);

% QPSK Gray mapper/demapper
modQPSK   = @(bits) ( (1-2*bits(1:2:end)) + 1i*(1-2*bits(2:2:end)) )/sqrt(2);
demodQPSK = @(z) [real(z)<0; imag(z)<0];  % returns 2 x Ns logical

% 파일럿 인덱스 균등 선택
make_pilots = @(NT_H,NT_V,Np) make_uniform_pilots(NT_H,NT_V,Np);

%% ---------------- Train one DNN per r (RESIDUAL) ----------------
nets = containers.Map('KeyType','double','ValueType','any');
stats = containers.Map('KeyType','double','ValueType','any');

for r = r_list
    Np = Npilot_dict(r);
    [P_idx, N_idx] = make_pilots(NT_H,NT_V,Np);

    inDim  = 2*NR*Np;
    outDim = 2*NR*(NT-Np);

    Xtr = zeros(N_train,inDim,'single');
    Ytr = zeros(N_train,outDim,'single');

    Es_train = 10^( (EbN0_train_dB+10*log10(bits_per_symbol))/10 );
    s2_tr = 1/Es_train;

    for n=1:N_train
        H = squeeze(gen_channel(1)); % (NR x NT)

        % partial pilots -> full-length LMMSE with NL averaging
        HfullP = zeros(NR,NT);
        for rx=1:NR
            yP = zeros(1,length(P_idx));
            for l=1:NL_train
                noise = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(s2_tr/2);
                yP = yP + (H(rx,P_idx) + noise);
            end
            yP = yP/NL_train;
            hhat = lmmse_from_partial(yP.', P_idx, s2_tr/NL_train);
            HfullP(rx,:) = hhat.';
        end

        Xin   = HfullP(:,P_idx);
        target_residual = H(:,N_idx) - HfullP(:,N_idx); % RESIDUAL

        xvec = [real(Xin(:)); imag(Xin(:))].';
        yvec = [real(target_residual(:)); imag(target_residual(:))].';
        Xtr(n,:) = single(xvec);
        Ytr(n,:) = single(yvec);

        % [학습 데이터 생성 for n=1:N_train 내부 맨 끝]
        if mod(n, log_every)==0
            fprintf('[TrainSet] %5d / %5d (%.1f%%)\n', n, N_train, 100*n/N_train);
        end

    end

    % Normalize X,Y
    muX = mean(Xtr,1); sX = std(Xtr,0,1)+1e-12;
    muY = mean(Ytr,1); sY = std(Ytr,0,1)+1e-12;
    XtrN = single((Xtr-muX)./sX); YtrN = single((Ytr-muY)./sY);

    % split
    ntr = floor((1-valFraction)*N_train);
    XTrain = XtrN(1:ntr,:);    YTrain = YtrN(1:ntr,:);
    XVal   = XtrN(ntr+1:end,:); YVal = YtrN(ntr+1:end,:);

    % DNN
    layers = [
        featureInputLayer(inDim,'Normalization','none')
        fullyConnectedLayer(hidden_units(1)); reluLayer
        fullyConnectedLayer(hidden_units(2)); reluLayer
        fullyConnectedLayer(hidden_units(3)); reluLayer
        fullyConnectedLayer(outDim)
        regressionLayer
    ];
    opts = trainingOptions('adam', ...
        'InitialLearnRate',learning_rate, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{XVal,YVal}, ...
        'Verbose',false, 'ExecutionEnvironment','auto');

    net = trainNetwork(XTrain,YTrain,layers,opts);

    nets(r)  = net;
    stats(r) = struct('P_idx',P_idx,'N_idx',N_idx,'muX',muX,'sX',sX,'muY',muY,'sY',sY);
    fprintf("DNN을 했다");
end

%% ---------------- BER simulation ----------------
BER_conv = zeros(numel(EbN0_dB),1);
BER_map  = containers.Map('KeyType','double','ValueType','any');
for r=r_list, BER_map(r)=zeros(numel(EbN0_dB),1); end

for si=1:numel(EbN0_dB)
    Es_lin = 10^(EsN0_dBs(si)/10);
    s2     = 1/Es_lin;

    % ---- Baseline(C): full-pilot LMMSE ----
    bitErr_C = 0; bitTot_C = 0;

    for tcase = 1:N_test
        H = squeeze(gen_channel(1)); % (NR x NT)

        % full-pilot with NL averaging
        Hc = zeros(NR,NT);
        for rx=1:NR
            y = zeros(1,NT);
            for l=1:NL_test
                n = (randn(1,NT)+1i*randn(1,NT))*sqrt(s2/2);
                y = y + (H(rx,:) + n);
            end
            y = y/NL_test;
            hhat = lmmse_full(y.', s2/NL_test);
            Hc(rx,:) = hhat.';
        end

        % Data transmission: one TX antenna at a time (TDMA/OFDMA)
        % Build Ns_data random QPSK symbols and random antenna indices
        bits = randi([0 1], 2*Ns_data, 1);
        s = modQPSK(bits).';               % 1 x Ns_data
        tx_idx = randi([1 NT], 1, Ns_data);

        for k=1:Ns_data
            tx = tx_idx(k);
            h  = H(:,tx);          % true (NR x 1)
            hh = Hc(:,tx);         % estimate (NR x 1)

            % Received vector y = h*s + n (NRx1)
            n = (randn(NR,1)+1i*randn(NR,1))*sqrt(s2/2);
            y = h*s(k) + n;

            % MRC combiner using estimate: g = hh/(||hh||^2+eps)
            g = hh/(norm(hh)^2 + 1e-12);
            rhat = g' * y;   % scalar

            % demap
            bhat = demodQPSK(rhat);
            bitErr_C = bitErr_C + sum(bhat(:) ~= bits(2*k-1:2*k));
            bitTot_C = bitTot_C + 2;
        end
    end
    BER_conv(si) = bitErr_C/bitTot_C;

    % ---- Proposed (for each r) ----
    for r = r_list
        cfg = stats(r); net = nets(r);
        P_idx = cfg.P_idx; N_idx = cfg.N_idx;
        muX=cfg.muX; sX=cfg.sX; muY=cfg.muY; sY=cfg.sY;

        bitErr = 0; bitTot = 0;

        for tcase = 1:N_test
            H = squeeze(gen_channel(1));

            % partial pilots -> LMMSE (full-length) with NL averaging
            HfullP = zeros(NR,NT);
            for rx=1:NR
                yP = zeros(1,length(P_idx));
                for l=1:NL_test
                    n = (randn(1,length(P_idx))+1i*randn(1,length(P_idx)))*sqrt(s2/2);
                    yP = yP + (H(rx,P_idx) + n);
                end
                yP = yP/NL_test;
                hhat = lmmse_from_partial(yP.', P_idx, s2/NL_test);
                HfullP(rx,:) = hhat.';
            end

            % DNN residual prediction for null antennas
            Xin = HfullP(:,P_idx);
            xin = [real(Xin(:)); imag(Xin(:))]';
            xinN = single((xin-muX)./sX);
            ypredN = predict(net, xinN);
            half = numel(ypredN)/2;
            res = (ypredN(1:half).*sY(1:half) + muY(1:half)) + ...
                1i*(ypredN(half+1:end).*sY(half+1:end) + muY(half+1:end));
            Res = reshape(res, NR, numel(N_idx));

            Hhat = HfullP;
            Hhat(:,N_idx) = HfullP(:,N_idx) + Res;

            % 데이터 구간
            bits = randi([0 1], 2*Ns_data, 1);
            s = modQPSK(bits).';
            tx_idx = randi([1 NT], 1, Ns_data);

            for k=1:Ns_data
                tx = tx_idx(k);
                h  = H(:,tx);     hh = Hhat(:,tx);
                n = (randn(NR,1)+1i*randn(NR,1))*sqrt(s2/2);
                y = h*s(k) + n;
                g = hh/(norm(hh)^2 + 1e-12);
                rhat = g' * y;
                bhat = demodQPSK(rhat);
                bitErr = bitErr + sum(bhat(:) ~= bits(2*k-1:2*k));
                bitTot = bitTot + 2;
            end
        end

        v = BER_map(r); v(si) = bitErr/bitTot; BER_map(r) = v; % Map 갱신(체인 인덱싱 X)
    end

    % 진행 출력 (Map에서 값 꺼내서 임시 변수로 인덱싱)
    v1 = BER_map(r_list(1)); v2 = BER_map(r_list(2)); v3 = BER_map(r_list(3));
    fprintf('SNR=%2d dB | BER(conv)=%.3e | proposed r=%.3f/%.3f/%.3f -> %.3e / %.3e / %.3e\n', ...
        EbN0_dB(si), BER_conv(si), r_list(1), r_list(2), r_list(3), v1(si), v2(si), v3(si));
end

%% ===== Throughput in bps with T = (1 - NP/200) * (1 - BER) * M * N_sc * Nsym_per_sec =====
% 시스템 시간/주파수 자원 (필요시 조정)
N_sc         = 1440;    % 사용 서브캐리어 개수 (예: 20 MHz 대역)
Nsym_per_sec = 16e3;    % 초당 OFDM 심볼 수 (예: 14 symbols/ms -> 14e3 sym/s)
M_bits       = 2;       % 변조 비트수: BPSK=1, QPSK=2, 16QAM=4 등
Denom        = 200;     % 식의 분모 200 (요구사항)

% --- Conventional (baseline): NP = NT (모든 안테나 파일럿) ---
NP_conv = NT;
data_frac_conv = max(0, 1 - NP_conv/Denom);
TH_conv_bps = data_frac_conv .* (1 - BER_conv) .* M_bits .* N_sc .* Nsym_per_sec;  % [bps]

% --- Proposed (각 r): NP = Npilot_dict(r) ---
TH_map_bps = containers.Map('KeyType','double','ValueType','any');
for r = r_list
    NP_r = Npilot_dict(r);
    data_frac_r = max(0, 1 - NP_r/Denom);
    BERv = BER_map(r);                       % 1 x numSNR
    TH_map_bps(r) = data_frac_r .* (1 - BERv) .* M_bits .* N_sc .* Nsym_per_sec;  % [bps]
end

% ---- Plot (bps) ----
figure('Color','w','Position',[100 100 900 600]);
plot(EbN0_dB, TH_conv_bps/1e7, '-o','LineWidth',1.5,'MarkerSize',6); hold on; % 보기 좋게 ×10^7로 스케일
mk = {'-^','->','-v'};
for i=1:numel(r_list)
    r = r_list(i);
    plot(EbN0_dB, TH_map_bps(r)/1e7, mk{i}, 'LineWidth',1.2,'MarkerSize',6);
end
grid on; box on;
xlabel('E_b/N_0 [dB]','FontSize',16);
ylabel('Throughput [\times10^7 bps]','FontSize',16);
title(sprintf('Throughput vs E_b/N_0 (N_T=%d, N_R=%d, M=%d)', NT, NR, M_bits),'FontSize',16);
legend({'Conventional method', ...
        sprintf('Proposed method (r = %.3f)', r_list(1)), ...
        sprintf('Proposed method (r = %.3f)', r_list(2)), ...
        sprintf('Proposed method (r = %.3f)', r_list(3))}, ...
        'Location','southeast','FontSize',12);




% %% ---------------- Plot (Fig.7 style) ----------------
% figure('Color','w','Position',[100 100 900 600]);
% semilogy(EbN0_dB, BER_conv,'-o','LineWidth',1.5,'MarkerSize',6); hold on;
% mk = {'-^','->','-v'};
% for i=1:numel(r_list)
%     r = r_list(i);
%     v = BER_map(r);
%     semilogy(EbN0_dB, v, mk{i}, 'LineWidth',1.2,'MarkerSize',6);
% end
% grid on; box on;
% xlabel('E_b/N_0 [dB]','FontSize',16);
% ylabel('BER','FontSize',16);
% title(sprintf('BER vs E_b/N_0 (N_T=%d, N_R=%d)',NT,NR),'FontSize',16);
% legend({'Conventional method', ...
%         sprintf('Proposed method (r = %.3f)',r_list(1)), ...
%         sprintf('Proposed method (r = %.3f)',r_list(2)), ...
%         sprintf('Proposed method (r = %.3f)',r_list(3))}, ...
%         'Location','southwest','FontSize',12);

%% ===== Local functions =====
function Hc = apply_corr(H, Rt_sqrt)
    [ns, NR, NT] = size(H);
    Hc = zeros(ns,NR,NT);
    for k=1:ns
        Hw = squeeze(H(k,:,:));     % (NR x NT)
        Hc(k,:,:) = Hw * Rt_sqrt;   % TX corr
    end
end

function H = gen_channel_peda(ns, NR, NT)
    % PedA PDP (평균 전력만 반영한 간단 모델) → flat-fading 등가화
    p_db  = [0 -1 -9 -10 -15 -20];
    p_lin = 10.^(p_db/10); p_lin = p_lin/sum(p_lin);
    H = zeros(ns,NR,NT);
    for k=1:ns
        Hk = zeros(NR,NT);
        for rx=1:NR
            for tx=1:NT
                taps = (randn(1,length(p_lin))+1i*randn(1,length(p_lin))).*sqrt(p_lin/2);
                Hk(rx,tx) = sum(taps);
            end
        end
        H(k,:,:) = Hk;
    end
end

function h_hat = lmmse_from_partial_fn(y_obs, P_idx, sigma2, Rt)
    P = P_idx(:).'; Rpp = Rt(P,P); Rtp = Rt(:,P);
    A = Rpp + sigma2*eye(length(P));
    w = A \ y_obs;
    h_hat = Rtp * w;                 % (NT x 1)
end

function [P_idx, N_idx] = make_uniform_pilots(NH,NV,Np)
    NHg = min(NH,Np);
    NVg = ceil(Np/NHg);
    if NVg>NV, NVg=NV; NHg = min(NH,ceil(Np/NVg)); end
    hs = unique(round(linspace(1,NH,NHg)),'stable');
    vs = unique(round(linspace(1,NV,NVg)),'stable');
    [HH,VV] = meshgrid(hs,vs);
    P = reshape((VV-1)*NH + HH,1,[]);
    if numel(P)>=Np
        pick = round(linspace(1,numel(P),Np));
        P_idx = sort(P(pick));
    else
        need = Np-numel(P); pool = setdiff(1:NH*NV,P);
        P_idx = sort([P, pool(round(linspace(1,numel(pool),need)))]);
    end
    N_idx = setdiff(1:NH*NV, P_idx);
end
