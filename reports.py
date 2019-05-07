import os
import json
import statistics


if __name__ == '__main__':
    ckpts_dir_list = list()
    for dirName, subdirList, fileList in os.walk('./'):
        if "results.txt" in fileList:
            print(dirName)
            ckpts_dir_list.append(dirName)

    results_dict = dict()
    for ckpts_dir in ckpts_dir_list:
        dataset_name, inliner_class, ckpt_datetime = tuple(ckpts_dir.split('/')[1:4])
        inliner_class = int(inliner_class)
        print(dataset_name, inliner_class, ckpt_datetime)

        results_file_path = os.path.join(ckpts_dir, 'results.txt')
        results_file = open(results_file_path)

        if dataset_name not in results_dict.keys():
            results_dict[dataset_name] = {}
        if inliner_class not in results_dict[dataset_name].keys():
            results_dict[dataset_name][inliner_class] = {}

        while True:
            # Get next line from file
            line = results_file.readline()
            # If line is empty then end of file reached
            if not line:
                break
            this_line = line.strip()
            if this_line == '':
                continue
            metric, score = tuple(this_line.split(':'))
            score = float(score)
            print(metric, score)

            if metric == 'Class':
                continue

            if metric not in results_dict[dataset_name][inliner_class].keys():
                results_dict[dataset_name][inliner_class][metric] = []

            results_dict[dataset_name][inliner_class][metric].append(score)

    for dataset_name in results_dict.keys():
        for inliner_class in results_dict[dataset_name].keys():
            for metric in results_dict[dataset_name][inliner_class].keys():
                results_dict[dataset_name][inliner_class][metric] = statistics.mean(results_dict[dataset_name][inliner_class][metric])

    with open('reports.json', 'w') as fp:
        json.dump(results_dict, fp, sort_keys=True, indent=4)
