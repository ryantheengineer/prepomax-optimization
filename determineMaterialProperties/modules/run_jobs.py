# -*- coding: utf-8 -*-
"""
Local Job Runner - Processes FEA jobs from local directories

@author: Ryan.Larson
"""

import yaml
import json
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path
import time
import uuid
import shutil

SETUP_CONFIG = "setup_config.yaml"

def get_process_id():
    """
    Gets a unique identifier for this process.
    """
    return f"local-{uuid.uuid4().hex[:8]}"

def claim_job(job_folder, process_id):
    """
    Atomically claim a job by creating a lock file.
    Returns True if job was successfully claimed, False if already claimed.
    """
    lock_file = job_folder / '.lock'
    claim_time = datetime.utcnow().isoformat()
    
    # Create lock content with process info for debugging
    lock_content = json.dumps({
        'claimed_by': process_id,
        'claimed_at': claim_time,
        'job_folder': str(job_folder)
    }, indent=2)
    
    try:
        # Try to create lock file exclusively (fails if file exists)
        # This is atomic on most filesystems
        with open(lock_file, 'x') as f:
            f.write(lock_content)
        
        print(f"✓ Successfully claimed job: {job_folder.name}")
        return True
        
    except FileExistsError:
        # Job already claimed by another process
        print(f"✗ Job already claimed: {job_folder.name}")
        
        # Optional: Show who claimed it (for debugging)
        try:
            with open(lock_file, 'r') as f:
                lock_info = json.loads(f.read())
            print(f"  Claimed by: {lock_info.get('claimed_by', 'unknown')} at {lock_info.get('claimed_at', 'unknown')}")
        except:
            pass  # Don't fail if we can't read the lock info
        
        return False
    except Exception as e:
        print(f"Error claiming job {job_folder.name}: {e}")
        raise

def release_job_lock(job_folder, process_id):
    """
    Release the job lock when processing fails.
    Call this only on failure - on success, keep the lock as a completion marker.
    """
    lock_file = job_folder / '.lock'
    
    try:
        # Get the lock to verify we own it
        if lock_file.exists():
            with open(lock_file, 'r') as f:
                lock_info = json.loads(f.read())
            
            if lock_info.get('claimed_by') == process_id:
                # We own the lock, safe to delete it
                lock_file.unlink()
                print(f"Released lock for job: {job_folder.name}")
            else:
                print(f"Warning: Cannot release lock for {job_folder.name} - owned by {lock_info.get('claimed_by')}")
        else:
            print(f"Lock for {job_folder.name} already released")
            
    except Exception as e:
        print(f"Error releasing lock for {job_folder.name}: {e}")

def list_unclaimed_jobs(jobs_root):
    """
    List only jobs that don't have lock files.
    Each job is a subdirectory containing a .yaml file.
    """
    jobs_root = Path(jobs_root)
    
    if not jobs_root.exists():
        print(f"Error: Jobs root directory does not exist: {jobs_root}")
        return []
    
    unclaimed_jobs = []
    
    # Iterate through subdirectories in the jobs root
    for item in jobs_root.iterdir():
        if item.is_dir():
            # Check if this directory has a .yaml file
            yaml_files = list(item.glob('*.yaml'))
            
            if yaml_files:
                # Check if there's no lock file (look for both .lock and *.lock patterns)
                lock_files = list(item.glob('*.lock'))
                if not lock_files:
                    unclaimed_jobs.append(item)
    
    return unclaimed_jobs

def create_error_file(job_folder, error_message, job_name=None):
    """
    Create an error file in the job folder to document job failures.
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        process_id = get_process_id()
        
        # Use job_name if available, otherwise use folder name
        if job_name is None:
            job_name = job_folder.name
        
        error_content = f"JOB PROCESSING ERROR\n"
        error_content += f"{'='*50}\n"
        error_content += f"Job: {job_name}\n"
        error_content += f"Job Folder: {job_folder}\n"
        error_content += f"Process ID: {process_id}\n"
        error_content += f"Error Time: {timestamp}\n"
        error_content += f"{'='*50}\n\n"
        error_content += f"ERROR DETAILS:\n{error_message}\n"
        
        # Create error file with timestamp to avoid overwrites
        error_filename = f"{job_name}_ERROR_{timestamp.replace(':', '-')}.txt"
        error_file = job_folder / error_filename
        
        # Write error file
        with open(error_file, 'w') as f:
            f.write(error_content)
        
        print(f"✓ Error file created: {error_file}")
        return error_file
        
    except Exception as e:
        print(f"✗ Failed to create error file: {e}")
        return None

def process_job(job_folder, process_id):
    """
    Process a single job with improved error handling and validation.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING JOB: {job_folder.name}")
    print(f"{'='*60}")
    
    # Find the YAML file in the job folder
    yaml_files = list(job_folder.glob('*.yaml'))
    
    if not yaml_files:
        raise Exception(f"No YAML file found in job folder: {job_folder}")
    
    if len(yaml_files) > 1:
        raise Exception(f"Multiple YAML files found in job folder: {job_folder}")
    
    job_yaml_path = yaml_files[0]
    
    print(f"Found job YAML: {job_yaml_path.name}")
    
    # Load and validate job configuration
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
    
    # List files in job folder
    print(f"\nFiles in job folder:")
    for fname in job_folder.iterdir():
        if fname.is_file():
            print(f"  {fname.name}")
    
    # Validate directories exist
    if not os.path.exists(opt_working_directory):
        raise Exception(f"Optimization working directory does not exist: {opt_working_directory}")
    
    if not os.path.exists(results_directory):
        print(f"Creating results directory: {results_directory}")
        os.makedirs(results_directory, exist_ok=True)
    
    # Run FEA + optimization with proper error handling
    print(f"\nRunning optimization with {job_yaml_path}")
    
    timeout_hours = 20
    timeout_seconds = timeout_hours * 3600
    
    try:
        result = subprocess.run(
            [sys.executable, 'runOptimizationCLI.py', str(job_yaml_path)],
            cwd=opt_working_directory,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        # result = subprocess.run(
        #     ['python', 'runOptimizationCLI.py', str(job_yaml_path)],
        #     cwd=opt_working_directory,
        #     capture_output=True,
        #     text=True,
        #     timeout=timeout_seconds
        # )
        
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
        raise Exception(f"Optimization script timed out after {timeout_hours} hours")
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
    
    # Copy results back to the job folder
    print(f"\nCopying results from {results_directory} to job folder...")
    
    if not os.path.exists(results_directory):
        raise Exception(f"Results directory does not exist: {results_directory}")
    
    filtered_extensions = (".png", ".result", ".log")  # Extensions that need job_name filtering
    all_extensions = (".dat", ".frd", ".txt", ".inp")   # Extensions to copy without filtering
    
    copied_count = 0
    for fname in os.listdir(results_directory):
        should_copy = False
        
        if fname.endswith(filtered_extensions):
            # For .png, .result, .log: only copy if filename contains job_name
            if job_name in fname:
                should_copy = True
                print(f"  ✓ File '{fname}' contains job name - will be copied")
            else:
                print(f"  ✗ File '{fname}' does not contain job name - skipping")
        elif fname.endswith(all_extensions):
            # For .dat, .frd, .txt, .inp: copy all files
            should_copy = True
            print(f"  ✓ File '{fname}' will be copied")
        
        if should_copy:
            try:
                src_path = os.path.join(results_directory, fname)
                dst_path = job_folder / fname
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                print(f"    Copied to: {dst_path}")
            except Exception as e:
                print(f"    ✗ Failed to copy {fname}: {e}")
    
    print(f"\n✓ Job processing complete. Copied {copied_count} files.")

def main_processing_loop(jobs_root):
    """
    Continuously process jobs until no more unclaimed jobs are available
    """
    jobs_root = Path(jobs_root)
    process_id = get_process_id()
    
    print("")
    print("#"*60)
    print(f"Process ID: {process_id}")
    print(f"Jobs Root: {jobs_root}")
    print("#"*60)
    
    jobs_processed = 0
    max_retries = 3  # Maximum number of cycles to try when no jobs can be claimed
    retry_count = 0
    
    while True:
        # Look for unclaimed jobs
        try:
            unclaimed_jobs = list_unclaimed_jobs(jobs_root)
        except Exception as e:
            print(f"✗ Error listing unclaimed jobs: {e}")
            break
        
        if not unclaimed_jobs:
            print(f"No more unclaimed jobs found. Total jobs processed: {jobs_processed}")
            break
        
        print(f"\nFound {len(unclaimed_jobs)} unclaimed job(s)")
        
        job_claimed_this_cycle = False
        
        for job_folder in unclaimed_jobs:
            print(f"\nAttempting to claim job: {job_folder.name}")
            
            try:
                if claim_job(job_folder, process_id):
                    try:
                        print(f"Processing job: {job_folder.name}")
                        process_job(job_folder, process_id)
                        
                        # Keep the lock file as a permanent record of completion
                        print(f"✓ Job completed successfully: {job_folder.name} (lock file retained)")
                        
                        jobs_processed += 1
                        job_claimed_this_cycle = True
                        retry_count = 0  # Reset retry count on successful job processing
                        break  # Process one job at a time, then check for more
                        
                    except Exception as e:
                        print(f"✗ Error processing job {job_folder.name}: {e}")
                        
                        # Create error file in the job folder
                        try:
                            job_name = None
                            yaml_files = list(job_folder.glob('*.yaml'))
                            if yaml_files:
                                try:
                                    with open(yaml_files[0], 'r') as f:
                                        job_config = yaml.safe_load(f)
                                        job_name = job_config.get("job_name")
                                except:
                                    pass
                            
                            error_file = create_error_file(job_folder, str(e), job_name)
                            if error_file:
                                print(f"✓ Error documented: {error_file}")
                        except Exception as error_file_exception:
                            print(f"✗ Failed to create error file: {error_file_exception}")
                        
                        # Release lock on failure so job can be retried
                        release_job_lock(job_folder, process_id)
                        
                        # Continue to try other jobs rather than stopping execution
                        print(f"↻ Continuing to next job...")
                        continue
                else:
                    print(f"Could not claim job: {job_folder.name}")
            except Exception as e:
                print(f"✗ Error in job claim/process cycle for {job_folder.name}: {e}")
                
                # Create error file for claim/process errors too
                try:
                    error_file = create_error_file(job_folder, f"Job claim/process error: {str(e)}")
                    if error_file:
                        print(f"✓ Claim/process error documented: {error_file}")
                except Exception as error_file_exception:
                    print(f"✗ Failed to create error file for claim error: {error_file_exception}")
                
                print(f"↻ Continuing to next job...")
                continue
        
        if not job_claimed_this_cycle:
            retry_count += 1
            print(f"No jobs could be claimed by this process in this cycle (attempt {retry_count}/{max_retries}).")
            
            if retry_count >= max_retries:
                print(f"Reached maximum retry attempts ({max_retries}). Stopping execution.")
                break
            
            # Wait a bit before checking again in case other processes are releasing jobs
            print("Waiting 30 seconds before checking for jobs again...")
            time.sleep(30)
    
    return jobs_processed > 0

# Main execution
if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python run_jobs_from_local.py <jobs_root_directory>")
    #     print("\nExample: python run_jobs_from_local.py ./jobs")
    #     sys.exit(1)
    
    # jobs_root = sys.argv[1]
    
    # Load and validate setup filepaths
    try:
        with open(SETUP_CONFIG, 'r') as f:
            setup_config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Failed to parse setup config YAML: {e}")
    
    jobs_root = setup_config["jobs_folder"]
    # jobs_root = "G:/Shared drives/RockWell Shared/Rockwell 3.0 Redesign Project/Strength + Performance/Flexural Stiffness Characterization/7 - Jobs"
    
    try:
        print("🚀 Starting job processing...")
        any_jobs_processed = main_processing_loop(jobs_root)
        
        if any_jobs_processed:
            print("✓ All available jobs have been processed.")
        else:
            print("ℹ No jobs were processed.")
        
        print("✅ Script execution completed.")
        
    except Exception as e:
        print(f"✗ Critical error in main execution: {e}")
        sys.exit(1)