
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
    current_path = os.path.dirname(os.path.abspath(__file__))

    yaml_conf = load_yaml_conf(yaml_file)
    # ps_ip = yaml_conf['ps_ip']
    ps_ip=socket.gethostname()
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        ip=socket.gethostname()
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))
    
    running_vms = set()
    subprocess_list=set()
    # job_name = 'kuiper_job'
    # log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""


    total_gpu_processes =  sum([sum(x) for x in total_gpus])
    learner_conf = '-'.join([str(_) for _ in list(range(1, total_gpu_processes+1))])

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        for item in yaml_conf['setup_commands']:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
    job_conf = {'time_stamp':time_stamp,
                'total_worker': total_gpu_processes,
                'ps_ip':ps_ip,
                'ps_port':random.randint(1000, 60000),
                'manager_port':random.randint(1000, 60000),
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    job_name = job_conf['job_name']
    if len(sys.argv)>3:
        job_conf['sample_mode'] = sys.argv[3]
    model_path = os.path.join(job_conf["log_path"], 'logs', job_name, time_stamp)
    job_conf["model_path"]=model_path
    
    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'


    log_file_name=os.path.join(current_path,f"{job_name}_logging_{time_stamp}") 
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    ps_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} "

    print(f"Starting time_stamp on {time_stamp}...")

    with open(log_file_name, 'wb') as fout:
        pass
    
    print(f"Starting aggregator on {ps_ip}...")
    with open(log_file_name, 'a') as fout:
        # p=subprocess.Popen(f'ssh -tt {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"', shell=True, stdout=fout, stderr=fout)
        
        # p=subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
        cmd_sequence=f'{ps_cmd}'
        cmd_sequence=cmd_sequence.split()
        p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)

        subprocess_list.add(p)
        time.sleep(30)

    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")
        for gpu_device  in range(len(gpu)):
            for _  in range(gpu[gpu_device]):
                # time.sleep(30)
                worker_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} --gpu_device={gpu_device+1}"
                rank_id += 1

                with open(log_file_name, 'a') as fout:
                    # p=subprocess.Popen(f'ssh -tt {submit_user}{worker} "{setup_cmd} {worker_cmd}"', shell=True, stdout=fout, stderr=fout)
                    
                    # p=subprocess.Popen(f'{worker_cmd}', shell=True, stdout=fout, stderr=fout)

                    cmd_sequence=f'{worker_cmd}'
                    cmd_sequence=cmd_sequence.split()
                    p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)

                    subprocess_list.add(p)

    exit_codes = [p.wait() for p in subprocess_list]

    # dump the address of running workers
    job_name = os.path.join(current_path, f"{job_name}_{time_stamp}")
    with open(job_name, 'wb') as fout:
        job_meta = {'user':submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs ({model_path}) for status")

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
    if len(sys.argv)==1:
        process_cmd('/mnt/home/lichenni/projects/Oort/training/evals/configs/speech/conf.yml')
    elif sys.argv[1] == 'submit':
        process_cmd(sys.argv[2])
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    else:
        print("Unknown cmds ...")
except:
    print("Error ...")

