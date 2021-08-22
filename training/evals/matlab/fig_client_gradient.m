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
    
    [val, time_index] = sort(completionTimes);
    val(1)
    val(100)
    time_index_short=time_index(1:100);
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
    set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 800 * 2, 450 * 2]);
    saveas(gcf, ['fig_clients_time.png'])

    total_data=[80*ones(size(dataSize));dataSize-80]';
    b=bar(total_data, 'stacked');
    for k = 1:size(total_data, 2)
        b(k).FaceColor = color_list(k,:);
    end

    legend({['Per-round Data Utility'],['Per-client Data Size']}, 'FontSize', 50, 'Position', [0.49 0.7 0.1 0.2], 'NumColumns', 1);
    xlabel('#ID of Clients'); % x label
    ylabel('Number of Data Samples'); % y label
    title('');
    % ylim([0, 22]);
    % set(gca, 'Ytick', 0:0.2:1)
    set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 800 * 2, 450 * 2]);
    saveas(gcf, ['fig_clients_data_utility.png'])

    error_path = [data_root, 'obs_local_epoch_gradient.mat'];
    a = load(error_path)
    error_matrix = struct2cell(a);

    gradient_l2_norm_list = error_matrix{1};
    gradient_l2_norm_list = cast(gradient_l2_norm_list, 'double');
    gradientUtilityList = error_matrix{2};
    gradientUtilityList = cast(gradientUtilityList, 'double');

    % yyaxis left
    % plot(completionTimesComm(time_index)./max(completionTimesComm),'LineWidth',4,'color',color_list(1,:));
    % hold on;
    % yyaxis right
    % plot(gradientUtilityList(time_index)./max(gradientUtilityList),'LineWidth',4,'color',color_list(2,:));

    total_data=[-completionTimesComm(time_index_short)./max(completionTimesComm); gradientUtilityList(time_index_short)./max(gradientUtilityList)]';
    bar(completionTimesComm(time_index_short)./max(completionTimesComm),0.3,'facecolor',color_list(1,:))
    hold on
    bar(-gradientUtilityList(time_index_short)./max(gradientUtilityList),0.3,'facecolor',color_list(2,:))

    % b=bar(total_data, 'stacked');
    % for k = 1:size(total_data, 2)
    %     b(k).FaceColor = color_list(k,:);
    % end

    legend({['Upload Communication'],['Per-update Importance']}, 'FontSize', 50, 'Position', [0.35 0.75 0.1 0.2], 'NumColumns', 1);
    xlabel('#ID of Clients'); % x label
    ylabel({'Comm. Time vs.Updates'}); % y label

    title('');
    % ylim([0, 22]);
    % set(gca, 'Ytick', 0:0.2:1)
    set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 800 * 2, 450 * 2]);
    saveas(gcf, ['fig_clients_gradient.png'])

end
