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
import sys
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

def create_error_file(job_key, s3, bucket, error_message, job_name=None):
    """
    Create an error file and upload it to S3 to document job failures.
    """
    try:
        job_folder = os.path.dirname(job_key)
        timestamp = datetime.utcnow().isoformat()
        instance_id = get_instance_id()
        
        # Use job_name if available, otherwise extract from job_key
        if job_name is None:
            job_name = os.path.basename(job_key).replace('.yaml', '')
        
        error_content = f"JOB PROCESSING ERROR\n"
        error_content += f"{'='*50}\n"
        error_content += f"Job: {job_name}\n"
        error_content += f"Job Key: {job_key}\n"
        error_content += f"Instance ID: {instance_id}\n"
        error_content += f"Error Time: {timestamp}\n"
        error_content += f"{'='*50}\n\n"
        error_content += f"ERROR DETAILS:\n{error_message}\n"
        
        # Create error file with timestamp to avoid overwrites
        error_filename = f"{job_name}_ERROR_{timestamp.replace(':', '-')}.txt"
        error_key = f"{job_folder}/{error_filename}"
        
        # Upload error file to S3
        s3.put_object(
            Bucket=bucket,
            Key=error_key,
            Body=error_content,
            ContentType='text/plain'
        )
        
        print(f"✓ Error file uploaded to S3: {error_key}")
        return error_key
        
    except Exception as e:
        print(f"✗ Failed to create/upload error file: {e}")
        return None

def process_job(job_key, s3, bucket):
    """
    Process a single job with improved error handling and validation.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING JOB: {job_key}")
    print(f"{'='*60}")
    
    # Download YAML and related files
    print("Downloading YAML and related files...")
    job_folder = os.path.dirname(job_key)
    local_job_dir = Path('/tmp/job')
    
    # Clean local job directory
    print(f"Cleaning local job directory: {local_job_dir}")
    if local_job_dir.exists():
        for f in local_job_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        local_job_dir.mkdir(parents=True, exist_ok=True)
        
    # Download all files in the job folder
    try:
        result = s3.list_objects_v2(Bucket=bucket, Prefix=job_folder + '/')
        for obj in result.get('Contents', []):
            key = obj['Key']
            filename = os.path.basename(key)
            local_path = local_job_dir / filename
            s3.download_file(bucket, key, str(local_path))
            print(f"  Downloaded: {filename}")
    except Exception as e:
        raise Exception(f"Failed to download job files: {e}")

    print(f"\nDownloaded job files:")
    for fname in local_job_dir.iterdir():
        if fname.is_file():
            print(f"  {fname.name}")
    
    # Load and validate job configuration
    job_yaml_path = local_job_dir / os.path.basename(job_key)
    
    if not job_yaml_path.exists():
        raise Exception(f"Job YAML file not found: {job_yaml_path}")
    
    try:
        with open(job_yaml_path, 'r') as f:
            job_config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Failed to parse job YAML: {e}")
        
    # Validate required fields
    required_fields = ["job_name", "results_directory", "opt_working_directory"]
    for field in required_fields:
        if field not in job_config:
            raise Exception(f"Missing required field in job config: {field}")
    
    job_name = job_config["job_name"]
    results_directory = job_config["results_directory"]
    opt_working_directory = job_config["opt_working_directory"]
    
    print(f'\nJOB NAME: {job_name}')
    print(f'RESULTS DIRECTORY: {results_directory}')
    print(f'WORKING DIRECTORY: {opt_working_directory}')
    
    # Validate directories exist
    if not os.path.exists(opt_working_directory):
        raise Exception(f"Optimization working directory does not exist: {opt_working_directory}")
    
    if not os.path.exists(results_directory):
        print(f"Creating results directory: {results_directory}")
        os.makedirs(results_directory, exist_ok=True)
    
    # Run FEA + optimization with proper error handling
    print(f"\nRunning optimization with {job_yaml_path}")
    
    try:
        result = subprocess.run(
            ['python', 'runOptimizationCLI.py', str(job_yaml_path)],
            cwd=opt_working_directory,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Check if the subprocess succeeded
        if result.returncode != 0:
            error_msg = f"Optimization script failed with return code {result.returncode}\n"
            error_msg += f"STDOUT: {result.stdout}\n"
            error_msg += f"STDERR: {result.stderr}"
            raise Exception(error_msg)
        
        print("✓ Optimization completed successfully")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        
    except subprocess.TimeoutExpired:
        raise Exception("Optimization script timed out after 1 hour")
    except Exception as e:
        raise Exception(f"Failed to run optimization script: {e}")
    
    # Verify expected output files were created
    expected_files = [
        f"{job_name}.result",
        f"{job_name}_optimization_plot.png"
    ]
    
    missing_files = []
    for expected_file in expected_files:
        expected_path = os.path.join(results_directory, expected_file)
        if not os.path.exists(expected_path):
            missing_files.append(expected_file)
    
    if missing_files:
        print(f"⚠ Warning: Expected output files not found: {missing_files}")
        # List what files are actually in the results directory
        print(f"Files in results directory {results_directory}:")
        if os.path.exists(results_directory):
            for fname in os.listdir(results_directory):
                print(f"  {fname}")
        else:
            print("  Directory does not exist!")
    
    # Upload results back to the job folder
    print(f"\nUploading results from {results_directory} to S3...")
    
    if not os.path.exists(results_directory):
        raise Exception(f"Results directory does not exist: {results_directory}")
    
    filtered_extensions = (".png", ".result", ".log")  # Extensions that need job_name filtering
    all_extensions = (".dat", ".frd", ".txt", ".inp")   # Extensions to upload without filtering
    
    uploaded_count = 0
    for fname in os.listdir(results_directory):
        should_upload = False
        
        if fname.endswith(filtered_extensions):
            # For .png, .result, .log: only upload if filename contains job_name
            if job_name in fname:
                should_upload = True
                print(f"  ✓ File '{fname}' contains job name - will be uploaded")
            else:
                print(f"  ✗ File '{fname}' does not contain job name - skipping")
        elif fname.endswith(all_extensions):
            # For .dat, .frd, .txt, .inp: upload all files
            should_upload = True
            print(f"  ✓ File '{fname}' will be uploaded")
        
        if should_upload:
            try:
                local_path = os.path.join(results_directory, fname)
                s3_key = f'{job_folder}/{fname}'
                s3.upload_file(local_path, bucket, s3_key)
                uploaded_count += 1
                print(f"    Uploaded to: s3://{bucket}/{s3_key}")
            except Exception as e:
                print(f"    ✗ Failed to upload {fname}: {e}")
    
    print(f"\n✓ Job processing complete. Uploaded {uploaded_count} files.")

def main_processing_loop():
    """
    Continuously process jobs until no more unclaimed jobs are available
    """
    s3 = boto3.client('s3')
    
    try:
        with open('s3_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ Failed to load s3_config.yaml: {e}")
        return False, None
        
    bucket = config['bucket']
    prefix = config['prefix']
    
    print("")
    print("#"*60)
    print(f"Instance ID: {get_instance_id()}")
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}")
    print("#"*60)
    
    jobs_processed = 0
    max_retries = 3  # Maximum number of cycles to try when no jobs can be claimed
    retry_count = 0
    
    while True:
        # Look for unclaimed jobs
        try:
            unclaimed_jobs = list_unclaimed_jobs(s3, bucket, prefix)
        except Exception as e:
            print(f"✗ Error listing unclaimed jobs: {e}")
            break
        
        if not unclaimed_jobs:
            print(f"No more unclaimed jobs found. Total jobs processed: {jobs_processed}")
            break
        
        print(f"\nFound {len(unclaimed_jobs)} unclaimed job(s)")
        
        job_claimed_this_cycle = False
        
        for job_key in unclaimed_jobs:
            print(f"\nAttempting to claim job: {job_key}")
            
            try:
                if claim_job(job_key, s3, bucket):
                    try:
                        print(f"Processing job: {job_key}")
                        process_job(job_key, s3, bucket)
                        
                        # Keep the lock file as a permanent record of completion
                        print(f"✓ Job completed successfully: {job_key} (lock file retained)")
                        
                        jobs_processed += 1
                        job_claimed_this_cycle = True
                        retry_count = 0  # Reset retry count on successful job processing
                        break  # Process one job at a time, then check for more
                        
                    except Exception as e:
                        print(f"✗ Error processing job {job_key}: {e}")
                        
                        # Create and upload error file to S3
                        try:
                            # Try to extract job_name from the error context or job file
                            job_name = None
                            try:
                                # Try to load the job file to get the job_name
                                local_job_dir = Path('/tmp/job')
                                job_yaml_path = local_job_dir / os.path.basename(job_key)
                                if job_yaml_path.exists():
                                    with open(job_yaml_path, 'r') as f:
                                        job_config = yaml.safe_load(f)
                                        job_name = job_config.get("job_name")
                            except:
                                pass  # If we can't get job_name, create_error_file will use a fallback
                            
                            error_file_key = create_error_file(job_key, s3, bucket, str(e), job_name)
                            if error_file_key:
                                print(f"✓ Error documented in S3: {error_file_key}")
                        except Exception as error_file_exception:
                            print(f"✗ Failed to create error file: {error_file_exception}")
                        
                        # Release lock on failure so job can be retried by another instance or later
                        release_job_lock(job_key, s3, bucket)
                        
                        # Continue to try other jobs rather than stopping execution
                        print(f"⏭ Continuing to next job...")
                        continue
                else:
                    print(f"Could not claim job: {job_key}")
            except Exception as e:
                print(f"✗ Error in job claim/process cycle for {job_key}: {e}")
                
                # Create and upload error file for claim/process errors too
                try:
                    error_file_key = create_error_file(job_key, s3, bucket, f"Job claim/process error: {str(e)}")
                    if error_file_key:
                        print(f"✓ Claim/process error documented in S3: {error_file_key}")
                except Exception as error_file_exception:
                    print(f"✗ Failed to create error file for claim error: {error_file_exception}")
                
                print(f"⏭ Continuing to next job...")
                continue
        
        if not job_claimed_this_cycle:
            retry_count += 1
            print(f"No jobs could be claimed by this instance in this cycle (attempt {retry_count}/{max_retries}).")
            
            if retry_count >= max_retries:
                print(f"Reached maximum retry attempts ({max_retries}). Stopping execution.")
                break
            
            # Wait a bit before checking again in case other instances are releasing jobs
            print("Waiting 30 seconds before checking for jobs again...")
            time.sleep(30)
    
    return jobs_processed > 0, config

def terminate_instance():
    try:
        with open('s3_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ Failed to load config for termination: {e}")
        return

    instance_id = get_instance_id()
    if instance_id.startswith("local-"):
        print(f"⚠ Skipping instance termination: Not running on EC2 (ID = {instance_id})")
        return

    try:
        print(f"Terminating instance: {instance_id}")
        ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2'))
        ec2.terminate_instances(InstanceIds=[instance_id])
        print("✓ Termination request sent successfully")
        time.sleep(30)  # Allow shutdown
    except Exception as e:
        print(f"✗ Error terminating instance: {e}")

def stop_instance():
    try:
        with open('s3_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ Failed to load config for stopping: {e}")
        return

    instance_id = get_instance_id()
    if instance_id.startswith("local-"):
        print(f"⚠ Skipping instance stop: Not running on EC2 (ID = {instance_id})")
        return

    try:
        print(f"Stopping instance: {instance_id}")
        ec2 = boto3.client('ec2', region_name=config.get('region', 'us-east-2'))
        ec2.stop_instances(InstanceIds=[instance_id])
        print("✓ Stop request sent successfully")
        time.sleep(30)  # Give it time to stop
    except Exception as e:
        print(f"✗ Error stopping instance: {e}")

# Updated main execution with proper error handling
if __name__ == "__main__":
    try:
        print("🚀 Starting job processing...")
        any_jobs_processed, config = main_processing_loop()
        
        if any_jobs_processed:
            print("✓ All available jobs have been processed. Initiating shutdown...")
        else:
            print("ℹ No jobs were processed. Initiating shutdown...")
        
    except Exception as e:
        print(f"✗ Critical error in main execution: {e}")
        print("🛑 Emergency shutdown initiated...")
    finally:
        # Ensure we always try to stop the instance
        print("🔄 Attempting to stop instance...")
        stop_instance()
        print("✅ Script execution completed.")