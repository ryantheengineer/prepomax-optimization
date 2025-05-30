# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:51:39 2025

@author: Ryan.Larson
"""

import boto3
import yaml
import os
import subprocess
import botocore.exceptions
from pathlib import Path
import requests
import time

s3 = boto3.client('s3')

with open('s3_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
bucket = config['bucket']
prefix = config['prefix']

print("")
print("#"*60)
print(f"Bucket:\t{bucket}")
print(f"Prefix:\t{prefix}")

def list_jobs():
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.yaml')]

# def claim_job(job_key):
#     # Implement a claim system (e.g., upload 'in_progress' marker)
#     try:
#         lock_key = job_key.replace('.yaml', '.lock')
#         s3.get_object(Bucket=bucket, Key=lock_key)
#         return False  # Already claimed
#     except:
#         s3.put_object(Bucket=bucket, Key=lock_key, Body=b'claimed')
#         return True


def claim_job(job_key):
    lock_key = job_key.replace('.yaml', '.lock')
    try:
        # Try to fetch the lock object
        s3.get_object(Bucket=bucket, Key=lock_key)
        return False  # Lock already exists
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            # Lock does not exist; create it
            s3.put_object(Bucket=bucket, Key=lock_key, Body=b'claimed')
            return True
        else:
            # Some other error occurred
            raise

def process_job(job_key):
    # Download YAML and related files
    print("Downloading YAML and related files...")
    job_folder = os.path.dirname(job_key)
    local_job_dir = Path('/tmp/job')
    
    # Clean local job directory
    print(f"Cleaning local job directory:\t{local_job_dir}")
    if os.path.exists(local_job_dir):
        for f in os.listdir(local_job_dir):
            os.remove(os.path.join(local_job_dir, f))
    else:
        os.makedirs(local_job_dir)
        
    # Download all files in the job folder
    result = s3.list_objects_v2(Bucket=bucket, Prefix=job_folder + '/')
    for obj in result.get('Contents', []):
        key = obj['Key']
        filename = os.path.basename(key)
        local_path = os.path.join(local_job_dir, filename)
        s3.download_file(bucket, key, local_path)

    print("Listing downloaded job files:")
    for fname in os.listdir(local_job_dir):
        print(f'\t{fname}')
    
    # Run FEA + optimization
    # job_yaml_path = os.path.join(local_job_dir, os.path.basename(job_key))
    job_yaml_path = local_job_dir / os.path.basename(job_key)
    job_yaml_path_str = str(job_yaml_path)
    
    with open(job_yaml_path_str, 'r') as f:
        job_config = yaml.safe_load(f)
        
    job_name = job_config["job_name"]
    results_directory = job_config["results_directory"]
    opt_working_directory = job_config["opt_working_directory"]
    
        
    print(f'WORKING ON JOB:\t{job_name}')
    print(f'RESULTS WILL BE SENT TO:\t{results_directory}')
    
    print(f"This is where the optimization will run with {job_yaml_path_str}")
    subprocess.run(
        ['python', 'runOptimizationCLI.py', job_yaml_path_str],
        cwd=opt_working_directory,
        text=True
    )
    
    # # Optionally save stdout/stderr to files
    # with open(os.path.join(local_job_dir, 'stdout.log'), 'w') as f:
    #     f.write(result.stdout)
    # with open(os.path.join(local_job_dir, 'stderr.log'), 'w') as f:
    #     f.write(result.stderr)
    
    # Upload results back to the job folder
    print("This is where the results will be uploaded back to S3")
    extensions = (".png", ".result", ".log", ".dat")
    for fname in os.listdir(results_directory):
        if fname.endswith(extensions):
            print(f"\tFile '{fname}' will be uploaded to S3")
    #     if fname.endswith('.result') or fname.endswith('.log') or fname.startswith('results'):
    #         s3.upload_file(
    #             os.path.join(results_directory, fname),
    #             bucket,
    #             f'{job_folder}/{fname}'
    #         )
    
def terminate_instance():
    try:
        # Get instance ID from metadata
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=5)
        instance_id = response.text
        print(f"Terminating instance: {instance_id}")
        
        ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2b'))
        ec2.terminate_instances(InstanceIds=[instance_id])
        
        # Sleep to allow shutdown
        time.sleep(30)
    except Exception as e:
        print(f"Error terminating instance: {e}")


for job_key in list_jobs():
    print(f"Current job key:\t{job_key}")
    if claim_job(job_key):
        process_job(job_key)
        break  # Stop after one job; instance shuts down or can loop
        
print("All results uploaded. Initiating shutdown...")
# terminate_instance()
