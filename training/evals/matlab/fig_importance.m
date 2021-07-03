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
set(fig,'defaultAxesColorOrder',[color_list(1,:); color_list(2,:)]);

date_str='0702_205637';
data_root = ['/mnt/home/lichenni/projects/Oort/training/evals/logs/google_speech/',date_str,'/worker/'];

error_path = [data_root, 'obs_importance.mat'];
a = load(error_path);
error_matrix = struct2cell(a);

timecost = error_matrix{2};
timecost = timecost(1:size(timecost,1), :)
size(timecost)

for ii = 1:size(timecost, 2)
    plot(timecost(:, ii), lineS{ii}, 'MarkerSize', 8, 'LineWidth', 4, 'color', color_list(ii, :));
    hold on;
end

legend({['Total Time'], ['Data Loader'], ['Gradient Norm'], ['Loss']}, 'FontSize', 40, 'Location', 'west','NumColumns',2);
xlabel('#Iterarion of Local Training'); % x label
ylabel('Time Cost'); % y label
ylim([0, 22]);
% set(gca, 'Ytick', 0:0.2:1)
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 360 * 2]);
saveas(gcf, ['fig_importance_time_',date_str,'.png'])
clf

epoch_selected = [1,5,10,20,30,40,50];
importance=error_matrix{1};

for jj = 1:length(epoch_selected)
    jj_index = epoch_selected(jj);

    yyaxis left
    data=squeeze(importance(:, jj_index, 1));
    plot(mean(data)*normalize(data,'range'), "-o", 'MarkerSize', 8, 'LineWidth', 4);
    hold on;

    yyaxis right
    data=squeeze(importance(:, jj_index, 2));
    plot(mean(data)*normalize(data,'range'), "--o", 'MarkerSize', 8, 'LineWidth', 4);

    % legend({['SGD Loss'],['Gradient Norm']}, 'FontSize', 40, 'Location', 'best');

    yyaxis left
    xlabel('#ID of Local Samples'); % x label
    ylabel('Scaled SGD Loss'); % y label
    yyaxis right
    ylabel('Scaled Gradient Norm'); % y label
    xlim([1, length(data)]);
    % set(gca, 'Ytick', 0:0.2:1)
    set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 360 * 2]);
    saveas(gcf, ['fig_importance_epoch', int2str(jj_index), '_',date_str,'.png'])
    clf
end
