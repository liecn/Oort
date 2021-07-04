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

set(fig, 'PaperSize', [6.8 4]);
% left_color = [.5 .5 0];
% right_color = [0 .5 .5];
% set(fig,'defaultAxesColorOrder',[color_list(1,:); color_list(2,:)]);

date_str='0702_205637';
data_root = ['/mnt/home/lichenni/projects/Oort/training/evals/logs/google_speech/',date_str,'/worker/'];

error_path = [data_root, 'obs_importance.mat'];
a = load(error_path);
error_matrix = struct2cell(a);

epoch_selected = [1,2,3];
importance=error_matrix{1};

for jj = 1:length(epoch_selected)
    jj_index = epoch_selected(jj);

    data=squeeze(importance(:, jj_index, 2));
    data=cast(data,'double');
    data=data/max(data)

    h=cdfplot(data);
    h.LineStyle=lineA(jj);
    h.Color=color_list(jj,:);
    h.Marker=lineC(jj);
    h.LineWidth=4;
    hold on
end
legend({['Google speech'],['OpenImg'], ['Stackoverflow']}, 'FontSize', 40, 'Location', 'southeast','NumColumns',1);

xlabel('Normalized Gradient Norm for each sample'); % y label
title('')
ylabel('CDF Across Samples');
% xlim([1, length(data)]);
% set(gca, 'Ytick', 0:0.2:1)
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 360 * 2]);
saveas(gcf, ['fig_sample.png'])
