# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:51:39 2025

@author: Ryan.Larson
"""

import boto3
import botocore.exceptions
import yaml
import requests
import json
from datetime import datetime
import os
import subprocess
from pathlib import Path
import time

def get_instance_id():
    """
    Gets the EC2 instance ID using IMDSv2.
    Falls back to a locally generated ID if not on EC2 or metadata is inaccessible.
    """
    try:
        # Step 1: Get a metadata token
        token_response = requests.put(
            'http://169.254.169.254/latest/api/token',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
            timeout=2
        )
        token_response.raise_for_status()
        token = token_response.text

        # Step 2: Use the token to get the instance ID
        metadata_response = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-id',
            headers={'X-aws-ec2-metadata-token': token},
            timeout=2
        )
        metadata_response.raise_for_status()
        return metadata_response.text.strip()

    except Exception as e:
        print(f"⚠ Warning: Failed to fetch EC2 metadata via IMDSv2: {e}. Falling back to local ID.")

    # Fallback for local development or failure
    import uuid
    fallback_id = f"local-{uuid.uuid4().hex[:8]}"
    print(f"⚠ Using fallback instance ID: {fallback_id}")
    return fallback_id



def claim_job(job_key, s3, bucket):
    """
    Atomically claim a job using S3 conditional operations.
    Returns True if job was successfully claimed, False if already claimed.
    """
    lock_key = job_key.replace('.yaml', '.lock')
    instance_id = get_instance_id()
    claim_time = datetime.utcnow().isoformat()
    
    # Create lock content with instance info for debugging
    lock_content = json.dumps({
        'claimed_by': instance_id,
        'claimed_at': claim_time,
        'job_key': job_key
    })
    
    try:
        # Use IfNoneMatch='*' to ensure the lock object doesn't exist
        # This operation is atomic - only one instance can succeed
        s3.put_object(
            Bucket=bucket,
            Key=lock_key,
            Body=lock_content,
            IfNoneMatch='*',  # Critical: only create if object doesn't exist
            ContentType='application/json'
        )
        
        print(f"✓ Successfully claimed job: {job_key}")
        return True
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code in ['ConditionalRequestFailed', 'PreconditionFailed']:
            # Job already claimed by another instance
            print(f"✗ Job already claimed: {job_key}")
            
            # Optional: Show who claimed it (for debugging)
            try:
                existing_lock = s3.get_object(Bucket=bucket, Key=lock_key)
                lock_info = json.loads(existing_lock['Body'].read())
                print(f"  Claimed by: {lock_info.get('claimed_by', 'unknown')} at {lock_info.get('claimed_at', 'unknown')}")
            except:
                pass  # Don't fail if we can't read the lock info
            
            return False
        else:
            # Some other error occurred
            print(f"Error claiming job {job_key}: {e}")
            raise

def release_job_lock(job_key, s3, bucket):
    """
    Release the job lock when processing is complete or fails.
    Call this in your process_job() function when done.
    """
    lock_key = job_key.replace('.yaml', '.lock')
    instance_id = get_instance_id()
    
    try:
        # Get the lock to verify we own it
        lock_obj = s3.get_object(Bucket=bucket, Key=lock_key)
        lock_info = json.loads(lock_obj['Body'].read())
        
        if lock_info.get('claimed_by') == instance_id:
            # We own the lock, safe to delete it
            s3.delete_object(Bucket=bucket, Key=lock_key)
            print(f"Released lock for job: {job_key}")
        else:
            print(f"Warning: Cannot release lock for {job_key} - owned by {lock_info.get('claimed_by')}")
            
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"Lock for {job_key} already released")
        else:
            print(f"Error releasing lock for {job_key}: {e}")

def list_unclaimed_jobs(s3, bucket, prefix):
    """
    List only jobs that don't have corresponding lock files.
    This is more efficient than checking locks individually.
    """
    # Get all objects in the prefix
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    all_objects = {obj['Key'] for obj in result.get('Contents', [])}
    
    # Find .yaml files that don't have corresponding .lock files
    unclaimed_jobs = []
    for key in all_objects:
        if key.endswith('.yaml'):
            lock_key = key.replace('.yaml', '.lock')
            if lock_key not in all_objects:
                unclaimed_jobs.append(key)
    
    return unclaimed_jobs

def process_job(job_key, s3, bucket):
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
    extensions = (".png", ".result", ".log", ".dat", ".frd", ".txt", ".inp")
    for fname in os.listdir(results_directory):
        if fname.endswith(extensions):
            print(f"\tFile '{fname}' will be uploaded to S3")
    #     if fname.endswith('.result') or fname.endswith('.log') or fname.startswith('results'):
            s3.upload_file(
                os.path.join(results_directory, fname),
                bucket,
                f'{job_folder}/{fname}'
            )

# Updated main processing loop
def main_processing_loop():
    """
    Updated main loop that handles job claiming more robustly
    """
    s3 = boto3.client('s3')
    
    with open('s3_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    bucket = config['bucket']
    prefix = config['prefix']
    
    print("")
    print("#"*60)
    print(f"Instance ID: {get_instance_id()}")
    print(f"Bucket:\t{bucket}")
    print(f"Prefix:\t{prefix}")
    print("#"*60)
    
    # Look for unclaimed jobs
    unclaimed_jobs = list_unclaimed_jobs(s3, bucket, prefix)
    
    if not unclaimed_jobs:
        print("No unclaimed jobs found.")
        return False, config
    
    print(f"Found {len(unclaimed_jobs)} unclaimed job(s)")
    
    # Try to claim jobs (shuffle for better distribution across instances)
    import random
    random.shuffle(unclaimed_jobs)
    
    for job_key in unclaimed_jobs:
        print(f"\nAttempting to claim job: {job_key}")
        
        if claim_job(job_key, s3, bucket):
            try:
                print(f"Processing job: {job_key}")
                process_job(job_key, s3, bucket)
                
                # Release the lock when done
                release_job_lock(job_key, s3, bucket)
                print(f"✓ Job completed successfully: {job_key}")
                return True, config
                
            except Exception as e:
                print(f"✗ Error processing job {job_key}: {e}")
                # Release lock on failure so job can be retried
                release_job_lock(job_key, s3, bucket)
                # Continue to try other jobs rather than failing completely
                continue
        else:
            print(f"Could not claim job: {job_key}")
    
    print("No jobs could be claimed by this instance.")
    return False, config

def terminate_instance():
    with open('s3_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    instance_id = get_instance_id()
    if instance_id.startswith("local-"):
        print(f"⚠ Skipping instance termination: Not running on EC2 (ID = {instance_id})")
        return

    try:
        print(f"Terminating instance: {instance_id}")
        ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2'))
        ec2.terminate_instances(InstanceIds=[instance_id])
        time.sleep(30)  # Allow shutdown
    except Exception as e:
        print(f"Error terminating instance: {e}")

        
def stop_instance():
    with open('s3_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    instance_id = get_instance_id()
    if instance_id.startswith("local-"):
        print(f"⚠ Skipping instance stop: Not running on EC2 (ID = {instance_id})")
        return

    try:
        print(f"Stopping instance: {instance_id}")
        ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2'))
        ec2.stop_instances(InstanceIds=[instance_id])
        time.sleep(30)  # Give it time to stop
    except Exception as e:
        print(f"Error stopping instance: {e}")


# Replace your existing main loop with:
if __name__ == "__main__":
    job_processed, config = main_processing_loop()
    
    if job_processed:
        print("All results uploaded. Initiating shutdown...")
        stop_instance()
    else:
        print("No jobs processed. Shutting down...")
        stop_instance()


# import boto3
# import yaml
# import os
# import subprocess
# import botocore.exceptions
# from pathlib import Path
# import requests
# import time


# def list_jobs(s3, bucket, prefix):
#     result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
#     return [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.yaml')]

# def claim_job(job_key, s3, bucket):
#     lock_key = job_key.replace('.yaml', '.lock')
#     try:
#         # Try to fetch the lock object
#         s3.get_object(Bucket=bucket, Key=lock_key)
#         return False  # Lock already exists
#     except botocore.exceptions.ClientError as e:
#         error_code = e.response['Error']['Code']
#         if error_code == 'NoSuchKey':
#             # Lock does not exist; create it
#             s3.put_object(Bucket=bucket, Key=lock_key, Body=b'claimed')
#             return True
#         else:
#             # Some other error occurred
#             raise

# def process_job(job_key, s3, bucket):
#     # Download YAML and related files
#     print("Downloading YAML and related files...")
#     job_folder = os.path.dirname(job_key)
#     local_job_dir = Path('/tmp/job')
    
#     # Clean local job directory
#     print(f"Cleaning local job directory:\t{local_job_dir}")
#     if os.path.exists(local_job_dir):
#         for f in os.listdir(local_job_dir):
#             os.remove(os.path.join(local_job_dir, f))
#     else:
#         os.makedirs(local_job_dir)
        
#     # Download all files in the job folder
#     result = s3.list_objects_v2(Bucket=bucket, Prefix=job_folder + '/')
#     for obj in result.get('Contents', []):
#         key = obj['Key']
#         filename = os.path.basename(key)
#         local_path = os.path.join(local_job_dir, filename)
#         s3.download_file(bucket, key, local_path)

#     print("Listing downloaded job files:")
#     for fname in os.listdir(local_job_dir):
#         print(f'\t{fname}')
    
#     # Run FEA + optimization
#     # job_yaml_path = os.path.join(local_job_dir, os.path.basename(job_key))
#     job_yaml_path = local_job_dir / os.path.basename(job_key)
#     job_yaml_path_str = str(job_yaml_path)
    
#     with open(job_yaml_path_str, 'r') as f:
#         job_config = yaml.safe_load(f)
        
#     job_name = job_config["job_name"]
#     results_directory = job_config["results_directory"]
#     opt_working_directory = job_config["opt_working_directory"]
    
        
#     print(f'WORKING ON JOB:\t{job_name}')
#     print(f'RESULTS WILL BE SENT TO:\t{results_directory}')
    
#     print(f"This is where the optimization will run with {job_yaml_path_str}")
#     subprocess.run(
#         ['python', 'runOptimizationCLI.py', job_yaml_path_str],
#         cwd=opt_working_directory,
#         text=True
#     )
    
#     # # Optionally save stdout/stderr to files
#     # with open(os.path.join(local_job_dir, 'stdout.log'), 'w') as f:
#     #     f.write(result.stdout)
#     # with open(os.path.join(local_job_dir, 'stderr.log'), 'w') as f:
#     #     f.write(result.stderr)
    
#     # Upload results back to the job folder
#     print("This is where the results will be uploaded back to S3")
#     extensions = (".png", ".result", ".log", ".dat", ".frd", ".txt", ".inp")
#     for fname in os.listdir(results_directory):
#         if fname.endswith(extensions):
#             print(f"\tFile '{fname}' will be uploaded to S3")
#     #     if fname.endswith('.result') or fname.endswith('.log') or fname.startswith('results'):
#             s3.upload_file(
#                 os.path.join(results_directory, fname),
#                 bucket,
#                 f'{job_folder}/{fname}'
#             )
    
# def terminate_instance(config):
#     try:
#         # Get instance ID from metadata
#         response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=5)
#         instance_id = response.text
#         print(f"Terminating instance: {instance_id}")
        
#         ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2b'))
#         ec2.terminate_instances(InstanceIds=[instance_id])
        
#         # Sleep to allow shutdown
#         time.sleep(30)
#     except Exception as e:
#         print(f"Error terminating instance: {e}")
        
# def stop_instance(config):
#     try:
#         # Get instance ID from metadata
#         response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=5)
#         instance_id = response.text
#         print(f"Stopping instance: {instance_id}")
        
#         ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2'))
#         ec2.stop_instances(InstanceIds=[instance_id])
        
#         # Sleep to allow shutdown
#         time.sleep(30)
#     except Exception as e:
#         print(f"Error stopping instance: {e}")


# def main_process():
#     s3 = boto3.client('s3')
    
#     with open('s3_config.yaml', 'r') as f:
#         config = yaml.safe_load(f)
        
#     bucket = config['bucket']
#     prefix = config['prefix']
    
#     print("")
#     print("#"*60)
#     print(f"Bucket:\t{bucket}")
#     print(f"Prefix:\t{prefix}")
    
#     for job_key in list_jobs(s3, bucket, prefix):
#         print(f"Current job key:\t{job_key}")
#         if claim_job(job_key, s3, bucket):
#             process_job(job_key, s3, bucket)
#             break  # Stop after one job; instance shuts down or can loop
            
#     print("All results uploaded. Initiating shutdown...")
#     stop_instance(config)
#     # terminate_instance(config)
    
# if __name__ == "__main__":
#     main_process()
