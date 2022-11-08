
% ---------------------------------------------------------------------- %

% addpath('../DeepMIMOv2/DeepMIMO_functions') % DeepMIMO functions folder
addpath(genpath('./DeepMIMOv2'))
addpath('./BF_codebook/')
params = read_params('MNALab_parameters.m');
[DeepMIMO_dataset, params] = DeepMIMO_generator(params);

%========================= Coordinated Deep Learning code ================
% Beamforming codebook parameters
over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
over_sampling_y=2;            % The beamsteering oversampling factor in the y direction
over_sampling_z=1;            % The beamsteering oversampling factor in the z direction

% Generating the beamforming codebook 
[BF_codebook]=UPA_codebook_generator(params.num_ant_BS(1), params.num_ant_BS(2), params.num_ant_BS(3), over_sampling_x, over_sampling_y, over_sampling_z, params.ant_spacing_BS);
codebook_size=size(BF_codebook,2);

num_sampled_OFDM=size(DeepMIMO_dataset{1}.user{1}.channel, 3);    % Number of OFDM samples which equals (from mmMIMO Dataset Generator) ofdm_num_subcarriers/output_subcarrier_downsampling_factor;

%%
% Adding noise
NF=5;             % Noise figure at the base station
Process_Gain=10;  % Channel estimation processing gain
BW=params.bandwidth*1e9; % System bandwidth in Hz
noise_power_dB=-204+10*log10(BW/params.num_OFDM)+NF-Process_Gain; % Noise power in dB
noise_power=10^(.1*(noise_power_dB)); % Noise power
num_antennas_tot=prod(params.num_ant_BS);

for u=1:1:params.num_user
   for t=1:length(params.active_BS)
        DeepMIMO_dataset{t}.user{u}.channel=DeepMIMO_dataset{t}.user{u}.channel+sqrt(noise_power)*(randn(1, num_antennas_tot, num_sampled_OFDM)+1j*randn(1, num_antennas_tot, num_sampled_OFDM));
   end
end

% Generating the DL inputs (the omni-received sequences)
DL_input_unnorm=zeros(params.num_user, length(params.active_BS)*num_sampled_OFDM);

for u=1:1:params.num_user
    for t=1:length(params.active_BS)
        DL_input_unnorm(u,(t-1)*num_sampled_OFDM+1:t*num_sampled_OFDM)=DeepMIMO_dataset{t}.user{u}.channel(1, 1,:);
    end 
end

% DL input normalization 
DL_input=DL_input_unnorm/max(max(abs(DL_input_unnorm)));
clear DL_input_unnorm;

% Generating the DL outputs (achievable rates of candidate/codebook beamforming vectors)
DL_output_unshaped=zeros(length(params.active_BS), params.num_user, codebook_size); 
for u=1:1:params.num_user
   for t=1:length(params.active_BS)
        Ch=squeeze(double(DeepMIMO_dataset{t}.user{u}.channel));
        DL_output_unshaped(t,u,:)=sum(log2(1+abs(Ch'*BF_codebook).^2),1)/num_sampled_OFDM;
        % DL output normalization
        max_Out=max(DL_output_unshaped(t,u,:));
        if max_Out ~= 0 
            DL_output_unshaped(t,u,:)=DL_output_unshaped(t,u,:)/max_Out;
        end
        [val, max_beam(t,u)]=max(DL_output_unshaped(t,u,:));
    end
end 

% Reshaping the output
DL_output=zeros(params.num_user, length(params.active_BS)*codebook_size); 
for t=1:length(params.active_BS)
    DL_output(:,(t-1)*codebook_size+1:t*codebook_size)=squeeze(DL_output_unshaped(t,:,:));
end

if ~exist('MNALab_DLBeam_dataset', 'dir')
    mkdir('MNALab_DLBeam_dataset')
end
save MNALab_DLBeam_Dataset/DLCB_input DL_input
save MNALab_DLBeam_Dataset/DLCB_output DL_output