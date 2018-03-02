#!/usr/bin/env python
import os
import sys

sys.path.append(os.path.dirname('../jdl_creator/')) # change path to classes directory
print sys.path
from classes.JDLCreator import JDLCreator  # import the class to create and submit JDL files

def main():
    """Submit a simple example job"""
    out_dir = "/storage/b/friese/toymass7/"
    jobs = JDLCreator("condocker")  # Default (no Cloud Site supplied): Docker with SLC6 image
    # Some example sites:
    # site_name='ekpsupermachines'  "Super Machines" IO intesiv jobs

    jobs.executable = "job.sh"  # name of the job script
    jobs.wall_time = 20 * 60 * 60  # job will finish in 10 hours
    jobs.memory = 2048  # Our regular 2048 MB per slot
    jobs.accounting_group = "cms.higgs"

    # build list of arguments: 1,2,3,4,5
    arguments = []
    #for masses in range(40,1000):
    for masses in range(50,1000):# range(40,1000):
        for seed in range(4):
            for channel in ["tt", "mt", "et", "em", "mm", "ee"]:
                for invert in ["true", "false"]:
                    arguments.append(" ".join([str(x) for x in [masses, seed, channel, invert, out_dir]]))
    print len(arguments)
    jobs.arguments = arguments  # set arguments for condor job

    # Our job requires lots of CPU resources and needs access to the local EKP resources
    jobs.requirements = "(TARGET.ProvidesCPU ==True) && (TARGET.ProvidesEKPResources == True)"

    jobs.job_folder = "condor_jobs"  # set name of the folder, where files and information are stored
    jobs.WriteJDL()  # write an JDL file and create folder for log files
    jobs.Submit()


if __name__ == "__main__":
    main()
