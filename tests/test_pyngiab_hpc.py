import os, time
import unittest
# local imports
from pyngiab_hpc import PyNGIABHPC_Anvil

class TestPyNGIAB(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def test_anvil_hpc_hello_world(self):
        from pyngiab_hpc import PyNGIABHPC_Anvil

        # [ToDo] There is a one time step involved here which will be documented

        hpc_client = PyNGIABHPC_Anvil(username="x-fbaig", allocation='cis220065')

        job_id = hpc_client.submit_job()
        print("Submitted job:", job_id)
        
        # Poll status
        while True:
            status = hpc_client.get_status(job_id)
            print("Status:", status)
                
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                break

            time.sleep(10)

        # Fetch output
        output = hpc_client.get_output(job_id)
        print(output)
        
        self.assertEqual(status, 'COMPLETED')
        pass

    def __del__(self):
        ''' Cleanup '''
        pass
