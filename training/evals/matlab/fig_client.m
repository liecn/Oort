% MAIN
% Version 30-Nov-2019
% Help on http://liecn.github.com
clear;
% clc;
close all;

%% Set Parameters for Loading Data
lineA = ["-", ":", "--", '-.'];
% lineB=[left_color,"b","m","g"];
lineC = ["*", "s", "o", "^", "+", "p", "d"];
lineS = ["-*", "--s", ":^", '-.p'];
color_list = linspecer(4, 'qualitative');

fig = figure;
set(fig, 'DefaultAxesFontSize', 40);
set(fig, 'DefaultAxesFontWeight', 'bold');

% set(fig, 'PaperSize', [6.8 4]);
set(fig,'defaultAxesColorOrder',[color_list(1,:); color_list(2,:)]);

date_str_list={'google_speech/0703_010524','google_speech/0703_003357','openimage/0703_160834','openimage/0703_150544'};
num_label=[35,35,596,596];
for ii=1:1:length(date_str_list)
    data_root = ['/mnt/home/lichenni/projects/Oort/training/evals/logs/',date_str_list{ii},'/worker/'];

    error_path = [data_root, 'obs_client.mat'];
    a = load(error_path)
    error_matrix = struct2cell(a);

    client_label = error_matrix{2};
    client_label=cast(client_label,'double');
    client_label=client_label/num_label(ii);

    h=cdfplot(client_label);
    h.LineStyle=lineA(ii);
    h.Color=color_list(ii,:);
    h.Marker=lineC(ii);
    h.LineWidth=4;
    % h.MarkerIndices=maker_idx;
    hold on;
end
legend({['Random speech'], ['Google speech'], ['Random openImg'],['OpenImg'],['Reddit']}, 'FontSize', 30, 'Location', 'southeast','NumColumns',1);
xlabel('Owned data with #kinds of Labels'); % x label
ylabel('CDF Across Clients'); % y label
title('');
% ylim([0, 22]);
% set(gca, 'Ytick', 0:0.2:1)
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 360 * 2]);
saveas(gcf, ['fig_clients_labels.png'])
clf

for ii=1:length(date_str_list)
    data_root = ['/mnt/home/lichenni/projects/Oort/training/evals/logs/',date_str_list{ii},'/worker/'];

    error_path = [data_root, 'obs_client.mat'];
    a = load(error_path);
    error_matrix = struct2cell(a);

    emd_distance = error_matrix{1};
    emd_distance=cast(emd_distance,'double');
    emd_distance=emd_distance*num_label(ii);

    h=cdfplot(emd_distance);
    h.LineStyle=lineA(ii);
    h.Color=color_list(ii,:);
    h.Marker=lineC(ii);
    h.LineWidth=4;
    % h.MarkerIndices=maker_idx;
    hold on;
end
legend({['Random speech'], ['Google speech'], ['Random openImg'],['OpenImg'], ['Reddit']}, 'FontSize', 30, 'Location', 'southeast','NumColumns',1);

xlabel('Scaled EMD divergence against the global distribution','FontSize', 35); % x label
ylabel('CDF Across Clients'); % y label
title('')
% ylim([0, 22]);
% set(gca, 'Ytick', 0:0.2:1)
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 360 * 2]);
saveas(gcf, ['fig_clients_emd.png'])
