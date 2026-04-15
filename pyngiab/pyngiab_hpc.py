import re
import time
import paramiko
import subprocess
from pathlib import Path

class PyNGIABHPC:
    def __init__(self, host, username, key_path, port=22):
        self.host = host
        self.username = username
        self.key_path = key_path
        self.port = port

        self._generate_ssh_keypair()

    def _generate_ssh_keypair(self):
        if not Path('/home/jovyan/.ssh/id_rsa.pub').exists():
            subprocess.run([
                "ssh-keygen",
                "-f", "/home/jovyan/.ssh/id_rsa",
                "-N", ""
            ])
        '''
        subprocess.run([
            "ssh-copy-id",
            "-i", "/home/jovyan/.ssh/id_rsa.pub"
            f'{self.username}@{self.host}'
        ])
        '''
        pass
    def _manual_setup_instructions(self):
        return ''' General setup instructions (will only work for HPC with password access)
        (1) Your public key is generated in /home/jovyan/.ssh/id_rsa.pub
        (2) In terminal run 'ssh-copy-id -i /home/jovyan/.ssh/id_rsa.pub user@hpc_host'
        (3) Put your password one time and you should be good to go
        '''

    def _connect(self):
        key = paramiko.RSAKey.from_private_key_file(self.key_path)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                pkey=key,
            )
            return client
        except Exception as e:
            raise ConnectionError('Unable to connect to remote host. Please follow instructions for one time setup.\n' + self._manual_setup_instructions())


    def submit_job(self, script_content=None):
        client = self._connect()

        if not script_content:
            script_content = self.create_job_script(command="echo 'PyNGIAB from Anvil!!!'")

        try:
            sftp = client.open_sftp()

            remote_script = f"/tmp/job_{int(time.time())}.sh"
            with sftp.file(remote_script, "w") as f:
                f.write(script_content)

            sftp.close()

            stdin, stdout, stderr = client.exec_command(f"sbatch {remote_script}")
            output = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                raise Exception(f"Submission error: {error}")

            # Extract job ID (Slurm format: "Submitted batch job 12345")
            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise Exception(f"Could not parse job ID: {output}")

            return match.group(1)

        finally:
            client.close()

    def get_status(self, job_id):
        client = self._connect()

        try:
            cmd = f"squeue -j {job_id} -h -o '%T'"
            stdin, stdout, stderr = client.exec_command(cmd)

            status = stdout.read().decode().strip()
            return status or "COMPLETED"

        finally:
            client.close()

    def get_output(self, job_id):
        client = self._connect()

        try:
            # Default Slurm output file pattern
            stdout_file = f"slurm-{job_id}.out"
            stdin, stdout, stderr = client.exec_command(f"cat {stdout_file}")

            return stdout.read().decode()

        finally:
            client.close()

    def create_job_script(self, command, cpus=2, mem="4G", time="00:15:00"):
        return f"""#!/bin/bash
#SBATCH --job-name=jupyter-job
#SBATCH --output=slurm-%j.out
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}

echo "Running on $(hostname)"
{command}
"""

class PyNGIABHPC_Anvil(PyNGIABHPC):
    def __init__(self,
                 username,
                 allocation,
                 key_path='/home/jovyan/.ssh/id_rsa',
                 port=22):
        host = 'anvil.rcac.purdue.edu'
        self._allocation=allocation
        super().__init__(host, username, key_path, port)
        pass

    def _manual_setup_instructions(self):
        return ''' Anvil setup instructions
        (1) Your public key is generated in /home/jovyan/.ssh/id_rsa.pub
        (2) Follow instructions at https://www.rcac.purdue.edu/knowledge/anvil/access/login/sshkeys
        '''
    def create_job_script(self, command, cpus=2, mem="4G", time="00:15:00"):
        return f"""#!/bin/bash
#SBATCH --job-name=pyngiab-job
#SBATCH --output=slurm-%j.out
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --account={self._allocation}

echo "Running on $(hostname)"
{command}
"""
