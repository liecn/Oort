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
color_list = linspecer(5, 'qualitative');

fig = figure;
set(fig, 'DefaultAxesFontSize', 50);
set(fig, 'DefaultAxesFontWeight', 'bold');

% set(fig, 'PaperSize', [6.8 4]);
set(fig, 'defaultAxesColorOrder', [color_list(1, :); color_list(2, :);]);

date_str_list = {'openimage/0811_093545_21190'};
num_label = [35, 35, 596, 596];

for ii = 1:1:length(date_str_list)
    data_root = ['/mnt/home/lichenni/projects/Oort/training/evals/logs/', date_str_list{ii}, '/aggregator/'];

    error_path = [data_root, 'obs_local_epoch_time.mat'];
    a = load(error_path)
    error_matrix = struct2cell(a);

    completionTimes = error_matrix{1};
    completionTimes = cast(completionTimes, 'double');
    completionTimesLocal = error_matrix{2};
    completionTimesLocal = cast(completionTimesLocal, 'double');
    completionTimesComm = error_matrix{3};
    completionTimesComm = cast(completionTimesComm, 'double');
    dataSizeRaw = error_matrix{4};
    dataSizeRaw = cast(dataSizeRaw, 'double');
    dataSizeRaw(23)=mean(dataSizeRaw);
    
    [~, time_index] = sort(completionTimes);
    time_index_short=time_index(1:100);

    dropout_ratio=0.1+0.004*[1:length(time_index_short)];
    completionTimesComm(time_index_short)=completionTimesComm(time_index_short).*(1-dropout_ratio);
    completionTimes(time_index_short)=completionTimesComm(time_index_short)+completionTimesLocal(time_index_short);
    completionTimes_std=max(completionTimes(time_index_short));
    completionTimesLocal(time_index_short)=(floor((completionTimes_std-completionTimes(time_index_short))./(completionTimesLocal(time_index_short)/5))/5+1).*completionTimesLocal(time_index_short);

    total_time = [completionTimesLocal(time_index); completionTimesComm(time_index)]'./max(completionTimes);
    dataSize = dataSizeRaw(time_index_short);

    b=bar(total_time, 'stacked');
    for k = 1:size(total_time, 2)
        b(k).FaceColor = color_list(k,:);
    end

    legend({['Local Computation'], ['Upload Communication']}, 'FontSize', 50, 'Location', 'northwest', 'NumColumns', 1);
    xlabel('#ID of Clients'); % x label
    ylabel('Scaled Time Consumption'); % y label
    title('');
    % ylim([0, 22]);
    % set(gca, 'Ytick', 0:0.2:1)
    set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 800 * 2, 600 * 2]);
    saveas(gcf, ['fig_clients_time_ours.png'])
end
