
# Submit job to the remote cluster

import yaml
import sys
import time
import random
import os, subprocess
import pickle, datetime

import socket

def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def process_cmd(yaml_file):

    yaml_conf = load_yaml_conf(yaml_file)
    # ps_ip = yaml_conf['ps_ip']
    ps_ip=socket.gethostname()
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    for ip_gpu in yaml_conf['worker_ips']:
        ip, num_gpu = ip_gpu.strip().split(':')
        ip=socket.gethostname()
        worker_ips.append(ip)
        total_gpus.append(int(num_gpu))

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    subprocess_list=set()
    # job_name = 'kuiper_job'
    # log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""


    job_conf = {'time_stamp':time_stamp,
                'total_worker': sum(total_gpus),
                'ps_ip':ps_ip,
                'ps_port':random.randint(1000, 60000),
                'manager_port':random.randint(1000, 60000)
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        for item in yaml_conf['setup_commands']:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "


    for conf_name in job_conf:
        if conf_name == "sample_mode" and len(sys.argv)>3:
            job_conf[conf_name] = sys.argv[3]
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(job_conf[conf_name], 'log', job_name, time_stamp)
        
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'

    learner_conf = '-'.join([str(_) for _ in list(range(1, sum(total_gpus)+1))])
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    ps_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} "

    print(f"Starting time_stamp on {time_stamp}...")

    with open(f"{job_name}_logging_{time_stamp}", 'wb') as fout:
        pass
    
    print(f"Starting aggregator on {ps_ip}...")
    with open(f"{job_name}_logging_{time_stamp}", 'a') as fout:
        # p=subprocess.Popen(f'ssh -tt {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"', shell=True, stdout=fout, stderr=fout)
        
        # p=subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
        cmd_sequence=f'{ps_cmd}'
        cmd_sequence=cmd_sequence.split()
        p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)

        subprocess_list.add(p)
        # time.sleep(10)

    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")
        for _  in range(gpu):
            time.sleep(30)

            worker_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} "
            rank_id += 1

            with open(f"{job_name}_logging_{time_stamp}", 'a') as fout:
                # p=subprocess.Popen(f'ssh -tt {submit_user}{worker} "{setup_cmd} {worker_cmd}"', shell=True, stdout=fout, stderr=fout)
                
                # p=subprocess.Popen(f'{worker_cmd}', shell=True, stdout=fout, stderr=fout)

                cmd_sequence=f'{worker_cmd}'
                cmd_sequence=cmd_sequence.split()
                p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)

                subprocess_list.add(p)

    exit_codes = [p.wait() for p in subprocess_list]

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, f"{job_name}_{time_stamp}")
    with open(job_name, 'wb') as fout:
        job_meta = {'user':submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs ({log_path}) for status")

def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
        print(f"Shutting down job on {vm_ip}")
        os.system(f"ssh {job_meta['user']}{vm_ip} '/mnt/home/lichenni/anaconda3/envs/oort/bin/python {current_path}/shutdown.py {job_name}'")
try:
    if sys.argv[1] == 'submit':
        process_cmd(sys.argv[2])
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    else:
        print("Unknown cmds ...")
except:
    print("Error ...")
    # process_cmd('/mnt/home/lichenni/projects/Oort/training/evals/configs/speech/conf.yml')

